# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./web"
from .dit_nodes import *
from .util_nodes import *
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PreViewVideo": PreViewVideo,
    "PABConfigNode": PABConfigNode,
    "DITPromptNode":DITPromptNode,
    "DITModelLoader":DITModelLoader,
    "DiffVAELoader":DiffVAELoader,
    "T5EncoderLoader":T5EncoderLoader,
    "T5TokenizerLoader": T5TokenizerLoader,
    "SchedulerLoader": SchedulerLoader,
    "LattePipeLineNode": LattePipeLineNode,
    "OpenSoraPlanPipeLineNode": OpenSoraPlanPipeLineNode,
    "OpenSoraNode": OpenSoraNode
}

'''
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Example Node"
}
'''
