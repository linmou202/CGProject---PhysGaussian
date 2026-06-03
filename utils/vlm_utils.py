import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from ..particle_filling.filling import get_aabb

def generate_bounded_image(
    file_name,
):
    # PART2_TODO: Use the aabb bounding box and camera position to edit the original image

    # Since you can do it in various ways (2D or 3D, etc.), feel free to pass in any variables 
    # available in gs_simulation before this function is called in gs_simulation.
    
    # this funciton returns nothing. Please save the generated image as "\generated_data\bounded_image\file_name"

    # if you want to use 3D gaussian rendering, refer to the lines after the comment "# run the simulation". It's basically the same.
    pass

def call_vlm(
    input_file_name,
    output_file_name
):
    # PART2_TODO: call the vlm, let it read "\generated_data\bounded_image\input_file_name"
    # and write the set of Young's Modulous into "\generated_data\vlm_data\output_file_name".
    
    # You can simply write 'pass' and do it manually.
    pass

def get_initial_params(
    file_name
):
    # PART2_TODO: Read the file "\generated_data\vlm_data\file_name"

    # Return a Tensor E consisting of Young's Modulous
    return E