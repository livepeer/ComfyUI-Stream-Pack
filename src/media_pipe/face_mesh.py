"""Source code for facial landmark detection and utilities using MediaPipe FaceMesh. 
Inspired by the ComfyUI ControlNet Auxiliary MediaPipe Face node.

References:
    ComfyUI ControlNet Auxiliary MediaPipe Face node - https://github.com/Fannovel16/comfyui_controlnet_aux
"""

import numpy as np
import cv2
import mediapipe as mp
import torch


class FaceMeshUtils:
    """
    Utility class for shared face mesh functionality.
    Contains drawing methods and specifications used by multiple nodes.
    """

    def __init__(self):
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # Define drawing specifications for different facial features
        self.f_thick = 2
        self.f_rad = 1
        self.right_iris_draw = self.mp_drawing.DrawingSpec(
            color=(10, 200, 250), thickness=self.f_thick, circle_radius=self.f_rad
        )
        self.right_eye_draw = self.mp_drawing.DrawingSpec(
            color=(10, 200, 180), thickness=self.f_thick, circle_radius=self.f_rad
        )
        self.right_eyebrow_draw = self.mp_drawing.DrawingSpec(
            color=(10, 220, 180), thickness=self.f_thick, circle_radius=self.f_rad
        )
        self.left_iris_draw = self.mp_drawing.DrawingSpec(
            color=(250, 200, 10), thickness=self.f_thick, circle_radius=self.f_rad
        )
        self.left_eye_draw = self.mp_drawing.DrawingSpec(
            color=(180, 200, 10), thickness=self.f_thick, circle_radius=self.f_rad
        )
        self.left_eyebrow_draw = self.mp_drawing.DrawingSpec(
            color=(180, 220, 10), thickness=self.f_thick, circle_radius=self.f_rad
        )
        self.mouth_draw = self.mp_drawing.DrawingSpec(
            color=(10, 180, 10), thickness=self.f_thick, circle_radius=self.f_rad
        )
        self.head_draw = self.mp_drawing.DrawingSpec(
            color=(10, 200, 10), thickness=self.f_thick, circle_radius=self.f_rad
        )

        # Create face connection specifications
        self.face_connection_spec = {}
        for edge in self.mp_face_mesh.FACEMESH_FACE_OVAL:
            self.face_connection_spec[edge] = self.head_draw
        for edge in self.mp_face_mesh.FACEMESH_LEFT_EYE:
            self.face_connection_spec[edge] = self.left_eye_draw
        for edge in self.mp_face_mesh.FACEMESH_LEFT_EYEBROW:
            self.face_connection_spec[edge] = self.left_eyebrow_draw
        for edge in self.mp_face_mesh.FACEMESH_RIGHT_EYE:
            self.face_connection_spec[edge] = self.right_eye_draw
        for edge in self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
            self.face_connection_spec[edge] = self.right_eyebrow_draw
        for edge in self.mp_face_mesh.FACEMESH_LIPS:
            self.face_connection_spec[edge] = self.mouth_draw

        # Iris landmark specification
        self.iris_landmark_spec = {468: self.right_iris_draw, 473: self.left_iris_draw}

    def reverse_channels(self, image):
        """Given a numpy array in RGB form, convert to BGR. Will also convert from BGR to RGB."""
        # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
        return image[:, :, ::-1]

    def draw_pupils(self, image, landmark_list, drawing_spec, halfwidth=2):
        """Draw pupils with custom function for better control"""
        if len(image.shape) != 3:
            raise ValueError("Input image must be H,W,C.")
        image_rows, image_cols, image_channels = image.shape
        if image_channels != 3:  # BGR channels
            raise ValueError("Input image must contain three channel bgr data.")

        # Ensure image is contiguous in memory for OpenCV compatibility
        image = np.ascontiguousarray(image, dtype=np.uint8)

        for idx, landmark in enumerate(landmark_list.landmark):
            if (landmark.HasField("visibility") and landmark.visibility < 0.9) or (
                landmark.HasField("presence") and landmark.presence < 0.5
            ):
                continue
            if (
                landmark.x >= 1.0
                or landmark.x < 0
                or landmark.y >= 1.0
                or landmark.y < 0
            ):
                continue
            image_x = int(image_cols * landmark.x)
            image_y = int(image_rows * landmark.y)
            draw_color = None
            if isinstance(drawing_spec, dict):
                if drawing_spec.get(idx) is None:
                    continue
                else:
                    draw_color = drawing_spec[idx].color
            elif isinstance(drawing_spec, self.mp_drawing.DrawingSpec):
                draw_color = drawing_spec.color
            image[
                image_y - halfwidth : image_y + halfwidth,
                image_x - halfwidth : image_x + halfwidth,
                :,
            ] = draw_color

    def draw_facemesh(self, img_rgb, results, thickness=1):
        """Draw face mesh visualization using the ControlNet Aux methodology on a black background"""
        # Create an empty image for drawing
        empty = np.zeros_like(img_rgb)

        # Ensure the image is contiguous in memory and has the correct data type for OpenCV
        empty = np.ascontiguousarray(empty, dtype=np.uint8)

        # Update thickness based on input parameter
        for spec in self.face_connection_spec.values():
            spec.thickness = thickness

        # Draw detected faces
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    empty,
                    face_landmarks,
                    connections=self.face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.face_connection_spec,
                )
                self.draw_pupils(empty, face_landmarks, self.iris_landmark_spec, 2)

        # Flip BGR back to RGB
        empty = self.reverse_channels(empty).copy()

        return empty

    def draw_facemesh_on_image(self, img_rgb, results, thickness=1):
        """Draw face mesh visualization on the provided image"""
        # Make a copy of the input image to avoid modifying the original
        img_copy = img_rgb.copy()

        # Convert RGB to BGR for OpenCV and MediaPipe drawing functions
        img_copy = self.reverse_channels(img_copy)

        # Ensure the image is contiguous in memory and has the correct data type for OpenCV
        img_copy = np.ascontiguousarray(img_copy, dtype=np.uint8)

        # Update thickness based on input parameter
        for spec in self.face_connection_spec.values():
            spec.thickness = thickness

        # Draw detected faces
        if results and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    img_copy,
                    face_landmarks,
                    connections=self.face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.face_connection_spec,
                )
                self.draw_pupils(img_copy, face_landmarks, self.iris_landmark_spec, 2)

        # Convert back from BGR to RGB
        img_copy = self.reverse_channels(img_copy).copy()

        return img_copy


# Create a global instance of the utils class
face_mesh_utils = FaceMeshUtils()


class FaceMeshDrawNode:
    """
    A node that draws MediaPipe face mesh on images.
    Takes face mesh data from FaceMeshDetectNode and visualizes it.
    Optimized for visualization only, requires an input image.
    Uses the same drawing methodology as ControlNet Aux for optimal performance.
    """

    def __init__(self):
        # Use the shared utils instance
        self.utils = face_mesh_utils

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "facemesh_data": ("FACEMESH_DATA",),
                "draw_contours": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "description": "Draw contour lines connecting mesh points",
                    },
                ),
                "mesh_thickness": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "display": "slider",
                        "description": "Thickness of mesh lines",
                    },
                ),
                "mesh_color": (
                    ["green", "red", "blue", "yellow", "white"],
                    {"default": "green", "description": "Color of the face mesh lines"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw_facemesh"
    CATEGORY = "ComfyUI_FaceMesh"

    def draw_facemesh(
        self, image, facemesh_data, draw_contours, mesh_thickness, mesh_color
    ):
        """Draw face mesh visualization on the input image"""
        # Get batch size
        batch_size = image.shape[0]
        output_images = []

        # Process each image in the batch
        for b in range(batch_size):
            # Get current image
            img = image[b].numpy()

            # Get corresponding facemesh data
            current_facemesh_data = facemesh_data[b]

            # If no face detected, add original image unchanged
            if not current_facemesh_data["face_detected"]:
                output_images.append(img)
                continue

            # Convert from float [0,1] to uint8 [0,255] for drawing
            img_rgb = (img * 255).astype(np.uint8)

            # Ensure the image is contiguous in memory for OpenCV compatibility
            img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

            # Use original results for drawing
            results = current_facemesh_data["original_results"]

            # Draw the face mesh on the input image using the shared utility
            output_img_rgb = self.utils.draw_facemesh_on_image(
                img_rgb, results, thickness=int(mesh_thickness)
            )

            # Convert back to float [0,1] for ComfyUI
            output_img = output_img_rgb.astype(np.float32) / 255.0

            # Add to output list
            output_images.append(output_img)

        # Convert output images list to tensor with batch dimension
        output_batch = torch.from_numpy(np.stack(output_images, axis=0))

        return (output_batch,)


class FaceMeshMaskNode:
    """
    Creates binary masks for different regions of the face detected by FaceMeshDetectNode.
    """

    # Define facial region indices
    FACIAL_REGIONS = {
        # These are approximate regions based on MediaPipe Face Mesh indices
        "whole_face": [
            # Face contour/oval - same as face_oval for proper face filling
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
            10,
        ],
        "mouth": [
            61,
            185,
            40,
            39,
            37,
            0,
            267,
            269,
            270,
            409,
            291,
            375,
            321,
            405,
            314,
            17,
            84,
            181,
            91,
            146,
        ],
        "left_eye": [
            # Left eye outline (from the person's perspective)
            263,
            249,
            390,
            373,
            374,
            380,
            381,
            382,
            362,
            263,
            466,
            388,
            387,
            386,
            385,
            384,
            398,
            362,
        ],
        "right_eye": [
            # Right eye outline
            33,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
            133,
            33,
            246,
            161,
            160,
            159,
            158,
            157,
            173,
            133,
        ],
        "nose": [
            # Nose area
            168,
            6,
            197,
            195,
            5,
            4,
            1,
            19,
            94,
            2,
            164,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            200,
            199,
            175,
        ],
        "left_eyebrow": [
            # Left eyebrow
            276,
            283,
            282,
            295,
            285,
            300,
            293,
            334,
            296,
            336,
        ],
        "right_eyebrow": [
            # Right eyebrow
            46,
            53,
            52,
            65,
            55,
            70,
            63,
            105,
            66,
            107,
        ],
        "face_oval": [
            # Face contour/oval
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
            10,
        ],
    }

    @classmethod
    def INPUT_TYPES(s):
        regions = list(s.FACIAL_REGIONS.keys())

        return {
            "required": {
                "facemesh_data": ("FACEMESH_DATA",),
                "face_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9,
                        "step": 1,
                        "display": "number",
                        "description": "Which face to mask (if multiple detected)",
                    },
                ),
                "region": (
                    regions,
                    {"default": "mouth", "description": "Facial region to mask"},
                ),
                "fill_region": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "description": "Fill the region or just draw contour",
                    },
                ),
                "mask_value": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "description": "Value/intensity of the mask (0.0-1.0)",
                    },
                ),
                "dilation_pixels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "display": "slider",
                        "description": "Dilate mask by this many pixels",
                    },
                ),
                "blur_radius": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 20,
                        "step": 1,
                        "display": "slider",
                        "description": "Apply gaussian blur with this radius",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "ComfyUI_FaceMesh"

    def create_mask(
        self,
        facemesh_data,
        face_index,
        region,
        fill_region,
        mask_value,
        dilation_pixels,
        blur_radius,
    ):
        """Create a mask for the specified facial region with custom intensity value"""
        # Get number of images in batch
        batch_size = len(facemesh_data)
        output_masks = []

        # Process each image in the batch
        for b in range(batch_size):
            # Get current facemesh data
            current_facemesh_data = facemesh_data[b]

            # If no face detected, return blank image
            if (
                not current_facemesh_data["face_detected"]
                or face_index >= current_facemesh_data["num_faces_detected"]
            ):
                # Create a blank mask with original image dimensions
                h, w = current_facemesh_data["image_shape"]
                mask_float = np.zeros((h, w), dtype=np.float32)
                output_masks.append(mask_float)
                continue

            # Get image shape
            h, w = current_facemesh_data["image_shape"]

            # Create blank mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # Get the landmarks for the specified face
            landmarks = current_facemesh_data["landmarks"][face_index]

            # Get indices for the specified region
            region_indices = self.FACIAL_REGIONS[region]

            # Extract points for the region - directly use numpy for better performance
            landmarks_array = np.array(landmarks)
            if len(region_indices) > 0:
                # Get subset of landmarks for the region
                region_landmarks = np.array(
                    [
                        landmarks_array[i]
                        for i in region_indices
                        if i < len(landmarks_array)
                    ]
                )

                if (
                    len(region_landmarks) > 2
                ):  # Need at least 3 points to create a contour
                    # Convert normalized coordinates to pixel coordinates
                    points = np.zeros((len(region_landmarks), 2), dtype=np.int32)
                    points[:, 0] = (region_landmarks[:, 0] * w).astype(
                        np.int32
                    )  # x coordinates
                    points[:, 1] = (region_landmarks[:, 1] * h).astype(
                        np.int32
                    )  # y coordinates

                    # Ensure mask is contiguous in memory for OpenCV compatibility
                    mask = np.ascontiguousarray(mask, dtype=np.uint8)

                    # Use 255 for drawing, we'll scale to mask_value later
                    fill_value = 255

                    # Special handling for mouth region - create a convex hull to ensure proper filling
                    if region == "mouth" and fill_region:
                        # Use convex hull to get a proper fillable shape
                        hull = cv2.convexHull(points)
                        cv2.fillConvexPoly(mask, hull, fill_value)
                    # For whole_face and face_oval, always fill regardless of fill_region setting
                    elif (
                        region == "whole_face" or region == "face_oval"
                    ) and fill_region:
                        # Use convex hull for better filling of the face
                        hull = cv2.convexHull(points)
                        cv2.fillConvexPoly(mask, hull, fill_value)
                    else:
                        # Create the mask based on fill_region option
                        if fill_region:
                            # Fill the polygon with white (255)
                            cv2.fillPoly(mask, [points], fill_value)
                        else:
                            # Just draw the outline with white (255)
                            cv2.polylines(mask, [points], True, fill_value, 2)

                    # Apply dilation if specified - use optimized kernel
                    if dilation_pixels > 0:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels)
                        )
                        mask = cv2.dilate(mask, kernel, iterations=1)

                    # Apply gaussian blur if specified
                    if blur_radius > 0:
                        mask = cv2.GaussianBlur(
                            mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0
                        )

            # Convert mask to float [0,1] for ComfyUI and apply mask_value
            mask_float = mask.astype(np.float32) / 255.0 * mask_value

            # Add to output list
            output_masks.append(mask_float)

        # Convert to batch tensor - BHW format
        mask_batch = torch.from_numpy(np.stack(output_masks, axis=0))

        return (mask_batch,)


class FaceMeshNode:
    """
    A node that uses MediaPipe to detect face mesh from images and draws it.
    Optimized for real-time video processing.
    Uses the same drawing methodology as ControlNet Aux for optimal performance.
    """

    def __init__(self):
        # Initialize face mesh detector once during node creation
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_results = None
        # Track current settings to know when to reinitialize
        self.current_settings = {
            "static_image_mode": False,
            "max_num_faces": 1,
            "refine_landmarks": False,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }
        # Use the shared utils instance
        self.utils = face_mesh_utils

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "static_image_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "description": "Set to true for static images, false for video streams (faster)",
                    },
                ),
                "max_num_faces": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "display": "number",
                        "description": "Maximum number of faces to detect (lower is faster)",
                    },
                ),
                "refine_landmarks": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "description": "Refine landmarks around eyes and lips (slower but more accurate)",
                    },
                ),
                "min_detection_confidence": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "description": "Minimum confidence for face detection",
                    },
                ),
                "min_tracking_confidence": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "description": "Minimum confidence for face tracking",
                    },
                ),
                "draw_mesh": (
                    "BOOLEAN",
                    {"default": True, "description": "Draw face mesh on output image"},
                ),
                "draw_contours": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "description": "Draw contour lines connecting mesh points",
                    },
                ),
                "mesh_color": (
                    ["green", "red", "blue", "yellow", "white"],
                    {"default": "green", "description": "Color of the face mesh lines"},
                ),
                "mesh_thickness": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "display": "slider",
                        "description": "Thickness of mesh lines",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FACEMESH_DATA", "IMAGE")
    RETURN_NAMES = ("facemesh_data", "annotations")
    FUNCTION = "detect_facemesh"
    CATEGORY = "ComfyUI_FaceMesh"

    def detect_facemesh(
        self,
        image,
        static_image_mode,
        max_num_faces,
        refine_landmarks,
        min_detection_confidence,
        min_tracking_confidence,
        draw_mesh,
        draw_contours,
        mesh_color,
        mesh_thickness,
    ):
        """Detect face mesh in the input image and optionally draw it using ControlNet Aux methodology"""
        # Check if any parameters have changed and we need to reinitialize
        new_settings = {
            "static_image_mode": static_image_mode,
            "max_num_faces": max_num_faces,
            "refine_landmarks": refine_landmarks,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }

        # Reinitialize if settings changed
        if new_settings != self.current_settings:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.current_settings = new_settings

        # Get batch size
        batch_size = image.shape[0]

        # Initialize empty lists to store batch results
        facemesh_data_batch = []
        output_images = []

        # Process each image in the batch
        for b in range(batch_size):
            # Get current image from batch
            img = image[b].numpy()

            # Convert from float [0,1] to uint8 [0,255] for mediapipe
            img_rgb = (img * 255).astype(np.uint8)

            # Ensure the image is contiguous in memory for OpenCV compatibility
            img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

            # Process the image
            results = self.face_mesh.process(img_rgb)

            # Prepare facemesh data structure for this image
            facemesh_data = {
                "landmarks": [],
                "face_detected": False,
                "num_faces_detected": 0,
                "tesselation": self.mp_face_mesh.FACEMESH_TESSELATION,
                "contours": self.mp_face_mesh.FACEMESH_CONTOURS,
                "image_shape": img.shape[:2],
                "original_results": results,  # Store original results for native drawing
            }

            # Process detection results
            if results and results.multi_face_landmarks:
                facemesh_data["face_detected"] = True
                facemesh_data["num_faces_detected"] = len(results.multi_face_landmarks)

                all_landmarks = []

                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Extract all landmarks at once for more efficiency
                    face_points = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                    all_landmarks.append(face_points)

                facemesh_data["landmarks"] = all_landmarks

            # Draw annotations if requested - on a black background for FaceMeshNode
            if draw_mesh and facemesh_data["face_detected"]:
                # Draw the face mesh on a black background using the shared utility
                empty = self.utils.draw_facemesh(
                    img_rgb, results, thickness=int(mesh_thickness)
                )

                # Convert back to float [0,1] for ComfyUI
                output_img = empty.astype(np.float32) / 255.0

                # Add image to output list
                output_images.append(output_img)
            else:
                # Use original image if no drawing requested or no face detected
                output_images.append(img)

            # Add facemesh data to batch results
            facemesh_data_batch.append(facemesh_data)

        # Convert output images list to tensor with batch dimension
        output_batch = torch.from_numpy(np.stack(output_images, axis=0))

        return (facemesh_data_batch, output_batch)
