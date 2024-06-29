import sys,os
import time
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
from torchvision.utils import save_image

from opendit.utils.utils import set_seed
from opendit.core.pab_mgr import set_pab_manager
from opendit.core.parallel_mgr import set_parallel_manager
from opendit.models.latte import LattePipeline,LatteT2V
from opendit.models.opensora_plan import LatteT2V as OpensoraPlanLatteT2V
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
                "model_type":(["latte","open_sora_plan","open_sora"],{
                    "default": "latte"
                })
            }
        }
    RETURN_TYPES = ("DITMODEL",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "get_model"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OpenDIT"

    def get_model(sef,model_type):
        if model_type == "latte":
            dit_local_dir = os.path.join(dit_models_dir,"Latte-1")
            snapshot_download(repo_id="maxin-cn/Latte-1",local_dir=dit_local_dir,ignore_patterns=["t2v_v20240523.pt"])
            transformer_model = LatteT2V.from_pretrained(dit_local_dir,subfolder="transformer").to(device,dtype=torch.float16)
        
        
        transformer_model.eval()
        return (transformer_model,)



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
                "num_inference_steps":("INT",{
                    "default": 50
                }),
                "guidance_scale":("FLOAT",{
                    "default": 7.5
                }),
                "enable_temporal_attentions":("BOOLEAN",{
                    "default": False
                }),
                "enable_vae_temporal_decoder":("BOOLEAN",{
                    "default": False
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
                  prompt,video_length,num_inference_steps,guidance_scale,
                  seed,enable_temporal_attentions,enable_vae_temporal_decoder,pab=None):
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
            enable_temporal_attentions=enable_temporal_attentions,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=enable_vae_temporal_decoder,
        ).video
        
        if sys.platform == "linux" and coordinator.is_master():
            if videos.shape[1] == 1:
                outfile = os.path.join(output_dir,prompt[:30].replace(" ", "_") + ".png")
                save_image(videos[0][0], outfile)
            else:
                outfile = os.path.join(output_dir,prompt[:30].replace(" ", "_") + "_%04d" % time.time_ns() + ".mp4")
                imageio.mimwrite(outfile,videos[0],fps=8,)
        else:
            if videos.shape[1] == 1:
                outfile = os.path.join(output_dir,prompt[:30].replace(" ", "_") + ".png")
                save_image(videos[0][0], outfile)
            else:
                outfile = os.path.join(output_dir,prompt[:30].replace(" ", "_") + "_%04d" % time.time_ns() + ".mp4")
                imageio.mimwrite(outfile,videos[0],fps=8,)
        return (outfile, )

        