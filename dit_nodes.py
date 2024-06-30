import sys,os
import time
import tqdm
from .util_nodes import now_dir,output_dir
sys.path.append(now_dir)
dit_models_dir = os.path.join(now_dir,"pretrained_models")
from huggingface_hub import snapshot_download

import torch
import imageio
import cuda_malloc
if sys.platform == "linux":
    import colossalai
    from colossalai.cluster import DistCoordinator
    from opendit.core.parallel_mgr import set_parallel_manager,enable_sequence_parallel

from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.schedulers import DDIMScheduler,PNDMScheduler
from diffusers.models import AutoencoderKLTemporalDecoder

from opendit.utils.utils import all_exists, create_logger, merge_args, set_seed, str_to_dtype
from opendit.core.pab_mgr import set_pab_manager
from opendit.models.latte import LattePipeline,LatteT2V
from opendit.models.opensora.datasets import ASPECT_RATIO_MAP,ASPECT_RATIOS,NUM_FRAMES_MAP,get_image_size,get_num_frames,save_sample
from opendit.models.opensora import STDiT3_XL_2,OpenSoraVAE_V1_2,T5Encoder,RFLOW,text_preprocessing
from opendit.models.opensora_plan import LatteT2V as OpensoraPlanLatteT2V,getae_wrapper,ae_stride_config,VideoGenPipeline
from opendit.models.opensora.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
device = f"cuda:{torch.cuda.current_device()}" if cuda_malloc.cuda_malloc_supported() else "cpu"

class PABConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps":("INT",{
                    "default": 100
                }),
                "cross_broadcast":("BOOLEAN",{
                    "default": False
                }),
                "cross_threshold":("STRING",{
                    "default": "100,900"
                }),
                "cross_gap":("INT",{
                    "default": 5
                }),
                "spatial_broadcast":("BOOLEAN",{
                    "default": False
                }),
                "spatial_threshold":("STRING",{
                    "default": "100,900"
                }),
                "spatial_gap":("INT",{
                    "default": 2
                }),
                "temporal_broadcast":("BOOLEAN",{
                    "default": False
                }),
                "temporal_threshold":("STRING",{
                    "default": "100,900"
                }),
                "temporal_gap":("INT",{
                    "default": 2
                })
            }
        }
    RETURN_TYPES = ("PABCONFIG",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_config"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_config(self,steps,cross_broadcast,cross_threshold,cross_gap,
                   spatial_broadcast,spatial_threshold,spatial_gap,
                   temporal_broadcast,temporal_threshold,temporal_gap):
        return {
            steps:steps,
            cross_broadcast:cross_broadcast,
            cross_threshold:[int(i) for i in cross_threshold.split(",")],
            cross_gap:cross_gap,
            spatial_broadcast:spatial_broadcast,
            spatial_threshold:[int(i) for i in spatial_threshold.split(",")],
            spatial_gap:spatial_gap,
            temporal_broadcast:temporal_broadcast,
            temporal_threshold:[int(i) for i in temporal_threshold.split(",")],
            temporal_gap:temporal_gap
        }

class DITPromptNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
             "required":{
                  "text": ("STRING", {
                      "multiline": True, 
                      "dynamicPrompts": True,
                      "default": "Time Lapse of the rising sun over a tree in an open rural landscape, with clouds in the blue sky beautifully playing with the rays of light",
                      })
             }
        }

    RETURN_TYPES = ("DITPROMPT",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_prompt"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_prompt(self,text):
        return (text,)
    
class DITModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "reuqired":{
                "model_type":(["open_sora","latte","open_sora_plan",],{
                    "default": "open_sora"
                })
            }
        }
    RETURN_TYPES = ("DITMODEL",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_model"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_model(sef,model_type,vae=None):
        if model_type == "latte":
            dit_local_dir = os.path.join(dit_models_dir,"Latte-1")
            snapshot_download(repo_id="maxin-cn/Latte-1",local_dir=dit_local_dir,ignore_patterns=["t2v_v20240523.pt"])
            transformer_model = LatteT2V.from_pretrained(dit_local_dir,
                                                         subfolder="transformer",
                                                         video_length=16).to(device,dtype=torch.float16)
            transformer_model.eval()
        elif model_type == "open_sora_plan":
            dit_local_dir = os.path.join(dit_models_dir,"Open-Sora-Plan-v1.1.0")
            snapshot_download(repo_id="LanguageBind/Open-Sora-Plan-v1.1.0",local_dir=dit_local_dir)
            transformer_model = OpensoraPlanLatteT2V.from_pretrained(dit_local_dir,subfolder="65x512x512",torch_dtype=torch.float16).to(device)
            transformer_model.eval()
        return (transformer_model,)

class DiffVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "reuqired":{
                "dit_type":(["open_sora","latte","open_sora_plan",],{
                    "default": "latte"
                })
            }
        }
    RETURN_TYPES = ("DIFFVAE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_vae"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_vae(sef,dit_type):
        if dit_type == "open_sora":
            vae_local_dir = os.path.join(dit_models_dir,"OpenSora-VAE-v1.2")
            snapshot_download(repo_id="hpcai-tech/OpenSora-VAE-v1.2",local_dir=vae_local_dir)
            vae = (
                OpenSoraVAE_V1_2(
                    from_pretrained=vae_local_dir,
                     micro_frame_size=17,
                    micro_batch_size=4,
                )
                .to(device,torch.float16)
                .eval()
            )
        elif dit_type == "latte":
            dit_local_dir = os.path.join(dit_models_dir,"Latte-1")
            snapshot_download(repo_id="maxin-cn/Latte-1",local_dir=dit_local_dir,ignore_patterns=["t2v_v20240523.pt"])
            vae = AutoencoderKLTemporalDecoder.from_pretrained(
                dit_local_dir,subfolder="vae_temporal_decoder", torch_dtype=torch.float16
            ).to(device)
            vae.eval()
        else:
            dit_local_dir = os.path.join(dit_models_dir,"Open-Sora-Plan-v1.1.0")
            snapshot_download(repo_id="LanguageBind/Open-Sora-Plan-v1.1.0",local_dir=dit_local_dir)
            vae = getae_wrapper("CausalVAEModel_4x8x8")(dit_local_dir,subfolder="vae").to(device, dtype=torch.float16)
            vae.vae.enable_tiling()
            vae.vae.tile_overlap_factor = 0.25
            vae.vae_scale_factor = ae_stride_config["CausalVAEModel_4x8x8"]
            vae.eval()

        return (vae,)

class T5EncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "reuqired":{
                "dit_type":(["open_sora","latte","open_sora_plan",],{
                    "default": "latte"
                })
            }
        }
    RETURN_TYPES = ("TEXTENCODER",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_t5"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_t5(sef,dit_type):
        if dit_type == "open_sora":
            t5_local_dir = os.path.join(dit_models_dir,"t5-v1_1-xxl")
            snapshot_download(repo_id="DeepFloyd/t5-v1_1-xxl",local_dir=t5_local_dir)
            text_encoder = T5Encoder(
                from_pretrained=t5_local_dir, model_max_length=300, device=device, shardformer=True
            )
        elif dit_type == "latte":
            dit_local_dir = os.path.join(dit_models_dir,"Latte-1")
            snapshot_download(repo_id="maxin-cn/Latte-1",local_dir=dit_local_dir,ignore_patterns=["t2v_v20240523.pt"])
            text_encoder = T5EncoderModel.from_pretrained(
                dit_local_dir, subfolder="text_encoder", torch_dtype=torch.float16
            ).to(device)
            text_encoder.eval()
        else:
            t5_local_dir = os.path.join(dit_models_dir,"t5-v1_1-xxl")
            snapshot_download(repo_id="DeepFloyd/t5-v1_1-xxl",local_dir=t5_local_dir)
            text_encoder = T5EncoderModel.from_pretrained(
                t5_local_dir, torch_dtype=torch.float16
            ).to(device)
            text_encoder.eval()

        return (text_encoder,)

class T5TokenizerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "reuqired":{
                "dit_type":(["latte","open_sora_plan",],{
                    "default": "latte"
                })
            }
        }
    RETURN_TYPES = ("TOKENIZER",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_tokenizer"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_t5(sef,dit_type):
        if dit_type == "latte":
            dit_local_dir = os.path.join(dit_models_dir,"Latte-1")
            snapshot_download(repo_id="maxin-cn/Latte-1",local_dir=dit_local_dir,ignore_patterns=["t2v_v20240523.pt"])
            tokenizer = T5Tokenizer.from_pretrained(dit_local_dir, subfolder="tokenizer")
        else:
            t5_local_dir = os.path.join(dit_models_dir,"t5-v1_1-xxl")
            snapshot_download(repo_id="DeepFloyd/t5-v1_1-xxl",local_dir=t5_local_dir)
            tokenizer = T5Tokenizer.from_pretrained(t5_local_dir)
        return (tokenizer,)

class SchedulerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "reuqired":{
                "dit_type":(["open_sora","latte","open_sora_plan",],{
                    "default": "latte"
                })
            }
        }
    RETURN_TYPES = ("SCHEDULER",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_scheduler"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_scheduler(sef,dit_type):
        if dit_type == "open_sora":
            scheduler = RFLOW(use_timestep_transform=True, num_sampling_steps=30, cfg_scale=7.0)
        elif dit_type == "latte":
            dit_local_dir = os.path.join(dit_models_dir,"Latte-1")
            snapshot_download(repo_id="maxin-cn/Latte-1",local_dir=dit_local_dir,ignore_patterns=["t2v_v20240523.pt"])
            scheduler = DDIMScheduler.from_pretrained(
                dit_local_dir,
                subfolder="scheduler",
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                variance_type="learned_range",
                clip_sample=False,
            )
        else:
            scheduler = PNDMScheduler()

        return (scheduler,)

class LattePipeLineNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model":("DITMODEL"),
                "vae":("DIFFVAE",),
                "text_encoder":("TEXTENCODER",),
                "tokenizer":("TOKENIZER",),
                "scheduler":("SCHEDULER",),
                "prompt":("DITPROMPT",),
                "video_length":("INT",{
                    "default": 16
                }),
                "fps":("INT",{
                    "default": 8
                }),
                "num_inference_steps":("INT",{
                    "default": 50
                }),
                "guidance_scale":("FLOAT",{
                    "default": 7.5
                }),
                "seed":("INT",{
                    "default": 42
                }),
            },
            "optional":{
                "pab":("PABCONFIG",)
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def gen_video(self,model,vae,text_encoder,tokenizer,scheduler,
                  prompt,video_length,fps,num_inference_steps,guidance_scale,
                  seed,pab=None):
        set_seed(seed)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # == init distributed env ==
        if sys.platform == "linux":
            colossalai.launch_from_torch({})
            coordinator = DistCoordinator()
            set_parallel_manager(1, coordinator.world_size)
        device = f"cuda:{torch.cuda.current_device()}"

        if pab:
            print(f"PAB params:\n{pab}")
            set_pab_manager(**pab)

        videogen_pipeline = LattePipeline(tokenizer=tokenizer,
                                          text_encoder=text_encoder,
                                          vae=vae,transformer=model,
                                          scheduler=scheduler).to(device)
        videos = videogen_pipeline(
            prompt,
            video_length=video_length,
            height=512,
            width=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            enable_temporal_attentions=True,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=True,
        ).video
        
        
        if videos.shape[1] == 1:
            outfile = os.path.join(output_dir,prompt[:30].replace(" ", "_") + ".png")
            save_image(videos[0][0], outfile)
        else:
            outfile = os.path.join(output_dir,prompt[:30].replace(" ", "_") + "_" + time.time_ns() + ".mp4")
            imageio.mimwrite(outfile,videos[0],fps=fps,)
        
        return (outfile, )

class OpenSoraPlanPipeLineNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model":("DITMODEL"),
                "vae":("DIFFVAE",),
                "text_encoder":("TEXTENCODER",),
                "tokenizer":("TOKENIZER",),
                "scheduler":("SCHEDULER",),
                "prompt":("DITPROMPT",),
                "num_frames":("INT",{
                    "default": 65
                }),
                "fps":("INT",{
                    "default": 24
                }),
                "num_inference_steps":("INT",{
                    "default": 150
                }),
                "guidance_scale":("FLOAT",{
                    "default": 7.5
                }),
                "seed":("INT",{
                    "default": 42
                }),
            },
            "optional":{
                "pab":("PABCONFIG",)
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def gen_video(self,model,vae,text_encoder,tokenizer,scheduler,
                  prompt,num_frames,fps,num_inference_steps,guidance_scale,
                  seed,pab=None):
        set_seed(seed)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # == init distributed env ==
        if sys.platform == "linux":
            colossalai.launch_from_torch({})
            coordinator = DistCoordinator()
            set_parallel_manager(1, coordinator.world_size)
        device = f"cuda:{torch.cuda.current_device()}"

        if pab:
            print(f"PAB params:\n{pab}")
            set_pab_manager(**pab)

        videogen_pipeline = VideoGenPipeline(tokenizer=tokenizer,
                                          text_encoder=text_encoder,
                                          vae=vae,transformer=model,
                                          scheduler=scheduler).to(device)
        videos = videogen_pipeline(
            prompt,
            num_frames=num_frames,
            height=512,
            width=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            enable_temporal_attentions=True,
            num_images_per_prompt=1,
            mask_feature=True,
        ).video
        outfile = os.path.join(output_dir, prompt[:30].replace(" ", "_")+ "_" + time.time_ns()+ ".mp4")
        imageio.mimwrite(
            outfile, videos[0], fps=fps, quality=9
        )  #
        
        return (outfile, )

class OpenSoraNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "vae":("DIFFVAE",),
                "text_encoder":("TEXTENCODER",),
                "tokenizer":("TOKENIZER",),
                "scheduler":("SCHEDULER",),
                "prompt":("DITPROMPT",),
                "resolution":(ASPECT_RATIO_MAP.keys(),{
                    "default": "480p"
                }),
                "aspect_ratio":(ASPECT_RATIOS.keys(),{
                    "default":"9:16"
                }),
                "num_frames":(NUM_FRAMES_MAP.keys(),{
                    "default": "2s"
                }),
                "fps":("INT",{
                    "default": 24
                }),
                "num_inference_steps":("INT",{
                    "default": 30
                }),
                "guidance_scale":("FLOAT",{
                    "default": 7.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
            },
            "optional":{
                "pab":("PABCONFIG",)
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_video(self,vae,text_encoder,tokenizer,scheduler,prompt,
                  resolution, aspect_ratio,num_frames,fps,
                  num_inference_steps,guidance_scale,seed,pab=None):
        torch.set_grad_enabled(False)
        # ======================================================
        # configs & runtime variables
        # ======================================================
        # == dtype ==
        dtype = torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if sys.platform == "linux":
            # == init distributed env ==
            colossalai.launch_from_torch({})
            coordinator = DistCoordinator()
            set_parallel_manager(1, coordinator.world_size)
            enable_sequence_parallelism = enable_sequence_parallel()
        device = f"cuda:{torch.cuda.current_device()}"
        set_seed(seed)

        # == init pab ==
        if pab:
            set_pab_manager(**pab)
        
        # == init logger ==
        logger = create_logger()
        verbose = 2
        progress_wrap = tqdm if verbose == 1 else (lambda x: x)
         # == prepare video size ==
        image_size = get_image_size(resolution, aspect_ratio)
        num_frames = get_num_frames(num_frames)
        input_size = (num_frames, *image_size)
        latent_size = vae.get_latent_size(input_size)
        # == build diffusion model ==
        local_dit_dir = os.path.join(dit_models_dir,"OpenSora-STDiT-v3")
        snapshot_download(repo_id="hpcai-tech/OpenSora-STDiT-v3",local_dir=local_dit_dir)
        model = (
            STDiT3_XL_2(
                from_pretrained=local_dit_dir,
                qk_norm=True,
                enable_flash_attn=True,
                enable_layernorm_kernel=True,
                input_size=latent_size,
                in_channels=vae.out_channels,
                caption_channels=text_encoder.output_dim,
                model_max_length=text_encoder.model_max_length,
            )
            .to(device, dtype)
            .eval()
        )
        text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

        # ======================================================
        # inference
        # ======================================================
        prompts = [prompt]
        reference_path = [""]
        mask_strategy = [""]

        # == prepare arguments ==
        save_fps = fps
        multi_resolution = "STDiT2"
        batch_size = 1
        num_sample = 1
        loop = 1
        condition_frame_length = 5
        condition_frame_edit = 0.0
        align = 5

        save_dir = os.path.join(output_dir, "open_sora")
        os.makedirs(save_dir, exist_ok=True)
        prompt_as_path = True

        # == Iter over all samples ==
        for i in progress_wrap(range(0, len(prompts), batch_size)):
            # == prepare batch prompts ==
            batch_prompts = prompts[i : i + batch_size]
            ms = mask_strategy[i : i + batch_size]
            refs = reference_path[i : i + batch_size]

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs = collect_references_batch(refs, vae, image_size)

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
            )

            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):
                # == prepare save paths ==
                save_paths = [
                    get_save_path_name(
                        save_dir,
                        sample_idx=idx,
                        prompt=original_batch_prompts[idx],
                        prompt_as_path=prompt_as_path,
                        num_sample=num_sample,
                        k=k,
                    )
                    for idx in range(len(batch_prompts))
                ]

                # NOTE: Skip if the sample already exists
                # This is useful for resuming sampling VBench
                if prompt_as_path and all_exists(save_paths):
                    continue

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                # 1. refine prompt by openai
                '''
                if args.llm_refine:
                    # only call openai API when
                    # 1. seq parallel is not enabled
                    # 2. seq parallel is enabled and the process is rank 0
                    if not enable_sequence_parallelism or (enable_sequence_parallelism and coordinator.is_master()):
                        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                            batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                    # sync the prompt if using seq parallel
                    if enable_sequence_parallelism:
                        coordinator.block_all()
                        prompt_segment_length = [
                            len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                        ]

                        # flatten the prompt segment list
                        batched_prompt_segment_list = [
                            prompt_segment
                            for prompt_segment_list in batched_prompt_segment_list
                            for prompt_segment in prompt_segment_list
                        ]

                        # create a list of size equal to world size
                        broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                        dist.broadcast_object_list(broadcast_obj_list, 0)

                        # recover the prompt list
                        batched_prompt_segment_list = []
                        segment_start_idx = 0
                        all_prompts = broadcast_obj_list[0]
                        for num_segment in prompt_segment_length:
                            batched_prompt_segment_list.append(
                                all_prompts[segment_start_idx : segment_start_idx + num_segment]
                            )
                            segment_start_idx += num_segment
                '''
                # 2. append score
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = append_score_to_prompts(
                        prompt_segment_list,
                        aes=6.5
                    )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    # == get prompt for loop i ==
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        refs, ms = append_generated(
                            vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )

                    # == sampling ==
                    z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    samples = scheduler.sample(
                        model,
                        text_encoder,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                    )
                    samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                    video_clips.append(samples)

                # == save samples ==
                if coordinator.is_master():
                    for idx, batch_prompt in enumerate(batch_prompts):
                        if verbose >= 2:
                            logger.info("Prompt: %s", batch_prompt)
                        save_path = save_paths[idx]
                        video = [video_clips[i][idx] for i in range(loop)]
                        for i in range(1, loop):
                            video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                        video = torch.cat(video, dim=1)
                        save_path = save_sample(
                            video,
                            fps=save_fps,
                            save_path=save_path,
                            verbose=verbose >= 2,
                        )
                        if save_path.endswith(".mp4") and args.watermark:
                            time.sleep(1)  # prevent loading previous generated video
                            add_watermark(save_path)
        logger.info("Inference finished.")
        logger.info("Saved samples to %s", save_dir)
        return (save_dir,)



