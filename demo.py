import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import random
from filter2noise import (
    prepare_image,
    DenoisingPipeline,
    LossFunction,
    restore_image
)
import pydicom
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

# Fix the seed for reproducibility
seed = 77
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


@dataclass
class DenoiseState:
    """
    Dataclass to hold the state of the interactive denoising process.

    Attributes:
        model (Optional[DenoisingPipeline]): The denoising model pipeline. Initialized in `initial_denoise`.
        device (Optional[str]): The device to run the model on ('cuda' if available, else 'cpu'). Determined in `initial_denoise`.
        tensor_image (Optional[torch.Tensor]): The input image as a torch tensor, prepared for the model. Created in `initial_denoise`.
        sigmas_list (Optional[List[torch.Tensor]]): List of sigma tensors output by each stage of the denoising model. Populated after initial denoising.
        current_stage_index (int): Index of the currently selected denoising stage (0-indexed). Defaults to 0 (first stage).
        current_sigma_index (int): Index of the currently selected sigma type (0: sigma_x, 1: sigma_y, 2: sigma_r). Defaults to 2 (sigma_r).
        original_sigma_maps (Dict[Tuple[int, int], np.ndarray]): Dictionary to store the original sigma maps after initial denoising.
                                                                 Keys are tuples (stage_index, sigma_index), values are numpy arrays.
    """
    model: Optional[DenoisingPipeline] = None
    device: Optional[str] = None
    tensor_image: Optional[torch.Tensor] = None
    sigmas_list: Optional[List[torch.Tensor]] = None
    current_stage_index: int = 0  # Track which stage we're modifying
    current_sigma_index: int = 2  # Default to sigma_r
    original_sigma_maps: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)  # Store original sigmas for visualization


class InteractiveDenoiseInterface:
    """
    Class for creating and managing the interactive Gradio interface for Filter2Noise denoising.

    This class handles image preprocessing, model initialization, denoising,
    sigma map manipulation, and user interface interactions.
    """
    def __init__(self):
        """
        Initializes the InteractiveDenoiseInterface with a DenoiseState, sigma names, and CSS styling.
        """
        self.state = DenoiseState()
        self.sigma_names = ["sigma_x", "sigma_y", "sigma_r"]
        self.state.original_sigma_maps = {} # Initialize the dictionary to store original sigma maps.
        self.css = """
        .gradio-container {
            max-width: 1400px;
            margin: auto;
            background-color: #f0f2f5;
            padding: 20px;
            border-radius: 8px;
        }
        .sigma-controls {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
        }
        .image-display {
            display: flex;
            justify-content: space-between;
            gap: 20px;
        }
        """

    def preprocess_image(self, input_image):
        """
        Preprocesses the input image to a normalized numpy array.

        Handles different input types: file paths (npy, dcm, ima) and numpy arrays.
        Normalizes the image to the range [0, 1] and converts it to float32.
        Grayscale conversion is performed if the input is a color image.

        Args:
            input_image (str or numpy.ndarray or PIL.Image.Image): Input image path, numpy array, or PIL Image.

        Returns:
            numpy.ndarray: Preprocessed image as a float32 numpy array in range [0, 1].

        Raises:
            ValueError: If an unsupported file format is encountered.
        """
        if isinstance(input_image, str):
            file_ext = os.path.splitext(input_image.lower())[1]
            if file_ext == '.npy':
                img_array = np.load(input_image).astype(np.float32)
            elif file_ext in ['.dcm', '.ima']:
                dicom = pydicom.dcmread(input_image)
                img_array = dicom.pixel_array.astype(np.float32)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        else:
            img_array = np.array(input_image).astype(np.float32)

        if len(img_array.shape) == 3:
            img_array = np.mean(img_array[:, :, :3], axis=2) # Convert color image to grayscale by averaging RGB channels.

        # Use min-max normalization
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > 1.0:
            img_array = img_array / 255.0
            img_min, img_max = img_array.min(), img_array.max()
        if img_max != img_min:
            img_array = (img_array - img_min) / (img_max - img_min)

        return img_array.astype(np.float32)

    def create_rainbow_colormap(self, sigma_array):
        """
        Generates a rainbow colormap (viridis) for visualizing sigma arrays.

        Uses a perceptually uniform colormap (viridis) and gamma correction
        to enhance visual distinction and highlight differences in sigma values.

        Args:
            sigma_array (numpy.ndarray): 2D numpy array representing sigma values.

        Returns:
            numpy.ndarray: RGB image (numpy array) representing the colormap visualization of the sigma array.
        """
        if sigma_array.min() == sigma_array.max():
            return np.zeros((*sigma_array.shape, 3), dtype=np.uint8) # Return black image if sigma array is constant.

        # Normalize sigma values to [0, 1]
        sigma_norm = (sigma_array - sigma_array.min()) / (sigma_array.max() - sigma_array.min())

        # Apply gamma correction for better visual contrast
        gamma = 0.5
        sigma_norm = sigma_norm ** gamma

        cmap = plt.cm.viridis  # Use perceptually uniform colormap (viridis)
        rgb_array = cmap(sigma_norm)[:, :, :3] # Convert normalized sigma to RGB using colormap
        rgb_array = (rgb_array * 255).astype(np.uint8) # Scale to [0, 255] and convert to uint8
        return rgb_array

    def create_overlay_colormap(self, original_sigma, updated_sigma, bbox, alpha=0.3):
        """
        Overlays the updated sigma map onto the original sigma map to visualize changes.

        Highlights the modified region defined by bbox on the original sigma map using alpha blending.
        This allows users to see the modified area in context with the original sigma distribution.

        Args:
            original_sigma (numpy.ndarray): Original sigma map numpy array.
            updated_sigma (numpy.ndarray): Updated sigma map numpy array (after adjustment in bbox).
            bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2) defining the modified region.
            alpha (float): Blending factor (0 to 1). Higher alpha means more updated sigma is visible.

        Returns:
            numpy.ndarray: RGB image (numpy array) of the overlaid colormap.
        """
        y1, x1, y2, x2 = bbox[1], bbox[0], bbox[3], bbox[2] # Unpack bounding box coordinates (top, left, bottom, right).
        original_map = self.create_rainbow_colormap(original_sigma) # Colormap for original sigma.
        updated_map = self.create_rainbow_colormap(updated_sigma)   # Colormap for updated sigma.

        # Extract the region defined by bbox from both original and updated colormaps.
        overlay_region_orig = original_map[y1:y2, x1:x2].astype(float)
        overlay_region_updated = updated_map[y1:y2, x1:x2].astype(float)

        # Alpha-blend the updated region onto the original region.
        blend_region = alpha * overlay_region_updated + (1.0 - alpha) * overlay_region_orig
        blend_region = blend_region.astype(np.uint8)

        # Place the blended region back into the original colormap.
        original_map[y1:y2, x1:x2] = blend_region
        return original_map

    def initial_denoise(self, input_file, alpha, epochs, learning_rate, num_stages, progress=gr.Progress()):
        """
        Initial denoising process: preprocesses image, trains the Filter2Noise model, and generates initial outputs.

        This function is triggered by the 'Denoise Image' button click.
        It performs the initial training of the Filter2Noise model on the uploaded image and
        generates the denoised image and sigma maps for the selected number of stages.

        Args:
            input_file (gradio.File): Uploaded input image file (Gradio File object).
            alpha (float): Lambda value (loss weight) for Filter2Noise training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the Adam optimizer.
            num_stages (int): Number of denoising stages in the Filter2Noise pipeline.
            progress (gradio.Progress): Gradio progress bar object to display training progress.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, gradio.Dropdown.update, gradio.Dropdown.update]:
            Tuple containing:
                - Input image (numpy array) for display.
                - Denoised output image (numpy array) for display.
                - Sigma map visualization (numpy array) for display (initial sigma_r of the first stage).
                - Gradio Dropdown update object for stage selector, populated with stage names.
                - Gradio Dropdown update object for sigma type selector, set to default sigma_r.
        """
        file_path = getattr(input_file, 'name', input_file) # Extract file path from Gradio File object.
        progress(0, desc="Preprocessing image...")
        processed_image = self.preprocess_image(file_path) # Preprocess the input image.
        input_display = (processed_image * 255).astype(np.uint8) # Prepare input image for display (scaled to 0-255).

        self.state.device = 'cuda' if torch.cuda.is_available() else 'cpu' # Determine device (CUDA if available, else CPU).
        self.state.tensor_image = prepare_image(processed_image, device=self.state.device) # Convert preprocessed image to tensor.

        progress(0.1, desc="Initializing model...")
        self.state.model = DenoisingPipeline(num_stages=num_stages, patch_size=8).to(self.state.device) # Initialize DenoisingPipeline model.
        loss_function = LossFunction(device=self.state.device, lambda_=alpha) # Initialize LossFunction.
        optimizer = torch.optim.Adam(self.state.model.parameters(), lr=learning_rate) # Initialize Adam optimizer.
        scheduler = torch.optim.lr_scheduler.OneCycleLR( # Initialize OneCycleLR scheduler.
            optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=1
        )

        self.state.model.train() # Set model to training mode.
        for epoch in range(epochs): # Training loop.
            optimizer.zero_grad() # Clear gradients.
            loss = loss_function(self.state.tensor_image, self.state.model) # Calculate loss.
            loss.backward() # Backpropagate gradients.
            optimizer.step() # Update model parameters.
            scheduler.step() # Update learning rate scheduler.

            progress((epoch + 1) / epochs, desc=f"Training: {epoch+1}/{epochs} epochs") # Update progress bar.

        progress(0.9, desc="Generating results...")

        with torch.no_grad(): # Disable gradient calculation for inference.
            denoised, self.state.sigmas_list = self.state.model(self.state.tensor_image, return_sigmas=True) # Run denoising and get sigma maps.

        output_array = restore_image(denoised) # Restore denoised tensor to numpy array.
        if output_array.max() <= 1.0:
            output_array = (output_array * 255).astype(np.uint8) # Scale to 0-255 for display if needed.

        # Initialize the current stage and sigma index for UI.
        self.state.current_stage_index = 0
        self.state.current_sigma_index = 2  # Default to sigma_r

        # Store original sigma maps for reset functionality and visualization.
        self.state.original_sigma_maps = {}
        for stage_idx, stage_sigmas in enumerate(self.state.sigmas_list):
            for sigma_idx in range(3):  # Iterate through sigma_x, sigma_y, sigma_r
                key = (stage_idx, sigma_idx)
                sigma_map = stage_sigmas[0, :, :, sigma_idx].cpu().numpy() # Extract sigma map from tensor and convert to numpy.
                self.state.original_sigma_maps[key] = sigma_map.copy() # Store a copy of the original sigma map.

        # Get the current sigma map for initial display (sigma_r of the first stage).
        current_sigma = self.state.sigmas_list[self.state.current_stage_index][0, :, :, self.state.current_sigma_index].cpu().numpy()
        sigma_display = self.create_rainbow_colormap(current_sigma) # Create colormap visualization for sigma map.

        input_display = input_display.squeeze() # Remove singleton dimensions for display.
        output_array = output_array.squeeze()

        # Convert grayscale images to RGB for Gradio Image component compatibility.
        if input_display.ndim == 2:
            input_display = np.stack([input_display] * 3, axis=-1)
        if output_array.ndim == 2:
            output_array = np.stack([output_array] * 3, axis=-1)

        return input_display, output_array, sigma_display, \
               gr.update(choices=[f"Stage {i+1}" for i in range(len(self.state.sigmas_list))], value=f"Stage 1"), \
               gr.update(choices=self.sigma_names, value=self.sigma_names[self.state.current_sigma_index]) # Return updates for Gradio components.

    def update_denoised_output_external(self):
        """
        Updates the denoised output image using the modified sigma maps.

        This function temporarily monkey-patches the forward pass of each stage's sigma predictor
        to use the externally updated sigma maps stored in `self.state.sigmas_list`.
        After generating the denoised image, it restores the original forward functions.

        Returns:
            numpy.ndarray: Updated denoised output image as a numpy array (scaled to 0-255 for display).
        """
        original_forwards = []
        # Monkey-patch each stage's sigma predictor forward function.
        for i, stage in enumerate(self.state.model.stages):
            sp = stage.sigma_predictor
            original_forwards.append(sp.forward) # Store original forward function.
            stage_idx = i
            # Replace forward function to return pre-computed sigma map from self.state.sigmas_list.
            sp.forward = lambda x, sp=sp, idx=stage_idx: self.state.sigmas_list[idx]

        with torch.no_grad(): # Disable gradient calculation for inference.
            denoised = self.state.model(self.state.tensor_image) # Run denoising with modified sigma maps.

        # Restore original forward functions for sigma predictors.
        for stage, orig in zip(self.state.model.stages, original_forwards):
            stage.sigma_predictor.forward = orig

        output_array = restore_image(denoised) # Restore denoised tensor to numpy array.
        if output_array.max() <= 1.0:
            output_array = (output_array * 255).astype(np.uint8) # Scale to 0-255 for display if needed.

        output_array = output_array.squeeze() # Remove singleton dimensions.
        if output_array.ndim == 2:
            output_array = np.stack([output_array] * 3, axis=-1) # Convert grayscale to RGB if needed.

        return output_array

    def adjust_sigma(self, bbox: Tuple[int, int, int, int], adjustment_factor: float):
        """
        Adjusts the sigma values within a selected bounding box region and updates the denoised output.

        Modifies the sigma map of the currently selected stage and sigma type based on the adjustment factor.
        Updates the denoised image based on the modified sigma maps and generates a visualization
        that overlays the modified sigma region on the original sigma map.

        Args:
            bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2) defining the region to adjust sigma values in.
            adjustment_factor (float): Factor to multiply the sigma values within the bbox by.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
            Tuple containing:
                - Updated denoised output image (numpy array) for display.
                - Sigma map visualization (numpy array) showing the adjusted region overlaid on the original sigma map.
        """
        stage_idx = self.state.current_stage_index # Get current stage index.
        sigma_idx = self.state.current_sigma_index # Get current sigma type index.

        # Get sigma map dimensions for bounding box clipping.
        sigma_height, sigma_width = self.state.sigmas_list[stage_idx][0, :, :, 0].shape
        x1, y1, x2, y2 = bbox # Unpack bounding box coordinates.
        x1 = max(0, int(x1)) # Clip x1 to be within image bounds.
        y1 = max(0, int(y1)) # Clip y1 to be within image bounds.
        x2 = min(sigma_width, int(x2)) # Clip x2 to be within image bounds.
        y2 = min(sigma_height, int(y2)) # Clip y2 to be within image bounds.

        mask = np.zeros((sigma_height, sigma_width), dtype=bool) # Create a mask for the bounding box region.
        mask[y1:y2, x1:x2] = True # Set mask to True within the bounding box.
        mask_tensor = torch.from_numpy(mask).to(self.state.device) # Convert mask to tensor and move to device.

        with torch.no_grad(): # Disable gradient calculation.
            stage_sigmas = self.state.sigmas_list[stage_idx].to(self.state.device) # Get current stage's sigma tensor.
            current_values = stage_sigmas[0, :, :, sigma_idx] # Get current sigma values for selected sigma type.
            adjustment = current_values * (adjustment_factor - 1.0) # Calculate the adjustment value.
            stage_sigmas[0, :, :, sigma_idx][mask] = current_values[mask] + adjustment[mask] # Apply adjustment to sigma values within the mask.
            stage_sigmas.clamp_(min=1e-6) # Clamp sigma values to a minimum value to prevent numerical issues.

        updated_output = self.update_denoised_output_external() # Update denoised output with modified sigma maps.
        updated_sigma = self.state.sigmas_list[stage_idx][0, :, :, sigma_idx].cpu().numpy() # Get updated sigma map as numpy array.

        # Get the original sigma map for overlay visualization.
        key = (stage_idx, sigma_idx)
        original_sigma_map = self.state.original_sigma_maps.get(key, updated_sigma.copy()) # Use updated sigma if original not found (shouldn't happen).

        # Create overlay colormap for visualization.
        sigma_display = self.create_overlay_colormap(
            original_sigma=original_sigma_map,
            updated_sigma=updated_sigma,
            bbox=(x1, y1, x2, y2),  # (left, top, right, bottom)
            alpha=0.5
        )

        return updated_output, sigma_display # Return updated output and sigma map visualization.

    def handle_selection(self, evt: gr.SelectData, adjustment_factor: float, stored):
        """
        Handles user selection events on the sigma map image for region adjustment.

        Manages bounding box selection through two clicks.
        - First click: Stores the starting point of the bounding box.
        - Second click: Completes the bounding box and triggers sigma adjustment in the selected region.

        Args:
            evt (gradio.SelectData): Gradio SelectData event containing click coordinates.
            adjustment_factor (float): Adjustment factor for sigma values.
            stored (Optional[Tuple[int, int]]): Stored coordinates of the first click (None if no click yet).

        Returns:
            Tuple[gradio.Image.update, gradio.Image.update, Optional[Tuple[int, int]]]:
            Tuple containing:
                - Gradio Image update object for the denoised output image.
                - Gradio Image update object for the sigma map visualization.
                - Updated stored click coordinates (None if selection completed, or first click coordinates).
        """
        if stored is None: # First click: store coordinates.
            if evt.index is not None and len(evt.index) == 2: # Check if click coordinates are valid.
                return gr.update(), gr.update(), evt.index # Store first click coordinates in state.
            else:
                return gr.update(), gr.update(), None # Invalid click, reset stored state.
        else: # Second click: complete selection and adjust sigma.
            if evt.index is not None and len(evt.index) == 2: # Check if second click coordinates are valid.
                x1 = min(stored[0], evt.index[0]) # Determine bounding box coordinates (top-left and bottom-right).
                y1 = min(stored[1], evt.index[1])
                x2 = max(stored[0], evt.index[0])
                y2 = max(stored[1], evt.index[1])
                bbox = (x1, y1, x2, y2) # Create bounding box tuple.
                updated_output, sigma_map_img = self.adjust_sigma(bbox, adjustment_factor) # Adjust sigma values in bbox.
                return updated_output, sigma_map_img, None # Return updated images and reset stored state.
            else:
                return gr.update(), gr.update(), stored # Invalid click, keep stored state.

    def change_stage(self, stage_choice):
        """
        Handles stage change events from the stage selector dropdown.

        Updates the current stage index in the state and refreshes the sigma map visualization
        to display the sigma map of the newly selected stage.

        Args:
            stage_choice (str): Selected stage name from the dropdown (e.g., "Stage 1").

        Returns:
            Tuple[numpy.ndarray, gradio.Dropdown.update]:
            Tuple containing:
                - Updated sigma map visualization (numpy array) for the new stage.
                - Gradio Dropdown update object for the sigma type selector (keeps current selection).
        """
        if not self.state.sigmas_list: # Return if sigma list is not initialized yet.
            return gr.update(), gr.update()

        stage_idx = int(stage_choice.split()[-1]) - 1 # Extract stage index from stage name.
        self.state.current_stage_index = stage_idx # Update current stage index in state.

        # Update the sigma map display for the new stage and current sigma type.
        current_sigma = self.state.sigmas_list[stage_idx][0, :, :, self.state.current_sigma_index].cpu().numpy()
        sigma_display = self.create_rainbow_colormap(current_sigma) # Create colormap visualization.

        return sigma_display, gr.update() # Return updated sigma map and keep sigma type selector as is.

    def change_sigma_type(self, sigma_choice):
        """
        Handles sigma type change events from the sigma type selector dropdown.

        Updates the current sigma type index in the state and refreshes the sigma map visualization
        to display the selected sigma type (sigma_x, sigma_y, or sigma_r) for the current stage.

        Args:
            sigma_choice (str): Selected sigma type name from the dropdown (e.g., "sigma_x").

        Returns:
            numpy.ndarray: Updated sigma map visualization (numpy array) for the new sigma type.
        """
        if not self.state.sigmas_list: # Return if sigma list is not initialized yet.
            return gr.update()

        sigma_idx = self.sigma_names.index(sigma_choice) # Get sigma index from sigma name.
        self.state.current_sigma_index = sigma_idx # Update current sigma index in state.

        # Update the sigma map display for the current stage and selected sigma type.
        current_sigma = self.state.sigmas_list[self.state.current_stage_index][0, :, :, sigma_idx].cpu().numpy()
        sigma_display = self.create_rainbow_colormap(current_sigma) # Create colormap visualization.

        return sigma_display # Return updated sigma map visualization.

    def reset_sigma_map(self):
        """
        Resets the sigma map of the current stage and sigma type to its initial state after training.

        Restores the original sigma map stored in `self.state.original_sigma_maps` and updates
        the denoised output image accordingly.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
            Tuple containing:
                - Updated denoised output image (numpy array) after resetting sigma map.
                - Updated sigma map visualization (numpy array) showing the reset sigma map.
        """
        if not self.state.sigmas_list or not self.state.original_sigma_maps: # Return if sigma maps not initialized or original maps not stored.
            return gr.update(), gr.update()

        stage_idx = self.state.current_stage_index # Get current stage index.
        sigma_idx = self.state.current_sigma_index # Get current sigma type index.
        key = (stage_idx, sigma_idx) # Create key for original sigma map dictionary.

        if key not in self.state.original_sigma_maps: # Return if original sigma map for current stage/sigma type not found.
            return gr.update(), gr.update()

        initial_sigma = self.state.original_sigma_maps[key] # Retrieve initial sigma map from stored originals.

        with torch.no_grad(): # Disable gradient calculation.
            initial_sigma_tensor = torch.from_numpy(initial_sigma).to(self.state.device) # Convert initial sigma to tensor and move to device.
            self.state.sigmas_list[stage_idx][0, :, :, sigma_idx] = initial_sigma_tensor # Update current sigma map with initial values.

        updated_output = self.update_denoised_output_external() # Update denoised output with reset sigma map.

        sigma_display = self.create_rainbow_colormap(initial_sigma) # Create colormap visualization for reset sigma map.

        return updated_output, sigma_display # Return updated output and sigma map visualization.

    def create_interface(self):
        """
        Creates the Gradio interface for the interactive denoising application.

        Defines the layout and components of the Gradio interface, including input file, parameters,
        buttons, image displays, and dropdown selectors for stage and sigma type.
        Sets up event listeners for user interactions with these components.

        Returns:
            gradio.Blocks: Gradio Blocks object representing the created interface.
        """
        with gr.Blocks(css=self.css) as interface: # Create Gradio Blocks interface with custom CSS.
            gr.Markdown("""
            # Interactive Filter2Noise
            Upload an image and then simulate a drag selection on the sigma map by clicking twice:
            The first click records the starting point and the second click records the ending point.
            The system will adjust the sigma values in the selected region and update the model
            to generate a new denoised image, while preserving the visible structure outside that region.
            """) # Markdown for interface description.

            with gr.Row(): # Row for input controls.
                with gr.Column(): # Column for input file and denoising parameters.
                    input_file = gr.File(label="Input Image", file_types=[".npy", ".dcm", ".ima"]) # File upload component.
                    alpha = gr.Number(value=350, label="lambda (Loss Weight)", precision=0) # Number input for lambda.
                    epochs = gr.Slider(minimum=100, maximum=3000, value=1000, step=100, label="Training Epochs") # Slider for epochs.
                    learning_rate = gr.Slider(minimum=1e-4, maximum=1e-2, value=1e-3, step=1e-4, label="Learning Rate") # Slider for learning rate.
                    num_stages = gr.Radio(choices=[1, 2], value=2, label="Number of Stages") # Radio buttons for number of stages.
                    denoise_button = gr.Button("Denoise Image") # Button to trigger initial denoising.

            with gr.Row(elem_classes=["image-display"]): # Row for image displays.
                input_image = gr.Image(label="Input Image", type="numpy") # Image component for input image.
                output_image = gr.Image(label="Denoised Result", type="numpy") # Image component for denoised output.
                sigma_map = gr.Image(label="Sigma Map (click to select points)", type="numpy", interactive=True) # Interactive Image component for sigma map.

            with gr.Row(elem_classes=["sigma-controls"]): # Row for sigma control elements.
                stage_selector = gr.Dropdown(label="Stage", choices=[], interactive=True) # Dropdown for stage selection.
                sigma_type_selector = gr.Dropdown(label="Sigma Type", choices=self.sigma_names, value=self.sigma_names[2], interactive=True) # Dropdown for sigma type selection.
                adjustment_factor = gr.Number(value=2.0, label="Adjustment Factor") # Number input for adjustment factor.
                reset_button = gr.Button("Reset Sigma Map") # Button to reset sigma map.

            select_state = gr.State(value=None) # Gradio State to store selection state for bounding box.

            # Event listeners for UI components.
            denoise_button.click(
                fn=self.initial_denoise,
                inputs=[input_file, alpha, epochs, learning_rate, num_stages],
                outputs=[input_image, output_image, sigma_map, stage_selector, sigma_type_selector]
            ) # Trigger initial denoising on button click.

            sigma_map.select(
                fn=self.handle_selection,
                inputs=[adjustment_factor, select_state],
                outputs=[output_image, sigma_map, select_state]
            ) # Handle sigma map selection for region adjustment.

            stage_selector.change(
                fn=self.change_stage,
                inputs=[stage_selector],
                outputs=[sigma_map, sigma_type_selector]
            ) # Handle stage change events.

            sigma_type_selector.change(
                fn=self.change_sigma_type,
                inputs=[sigma_type_selector],
                outputs=[sigma_map]
            ) # Handle sigma type change events.

            reset_button.click(
                fn=self.reset_sigma_map,
                inputs=[],
                outputs=[output_image, sigma_map]
            ) # Handle reset sigma map button click.

            return interface # Return the created Gradio interface.

if __name__ == "__main__":
    app = InteractiveDenoiseInterface() # Create instance of the interface class.
    interface = app.create_interface() # Create the Gradio interface.
    interface.launch(share=False) # Set share=True to enable public sharing.
