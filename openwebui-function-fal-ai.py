"""
title: Fal.ai Master Image Generator
description: A unified pipe to generate images using various Fal.ai models. Use the 'Custom' option to test new models defined in the MODEL_ID Valve.
author: User
version: 1.1.0
license: MIT
requirements: fal-client, python-dotenv
environment_variables: FAL_KEY
"""

import os
import asyncio
from typing import List, Callable, Awaitable
from pydantic import BaseModel, Field
import fal_client


class Pipe:
    class Valves(BaseModel):
        FAL_KEY: str = Field(default="", description="API Key for Fal.ai (required)")
        MODEL_ID: str = Field(
            default="fal-ai/flux-pro/v1.1-ultra",
            description="Default/Custom Model ID. Used ONLY if you select 'Fal.ai Custom' or if the selected model is not found.",
        )
        # Dimensions for models that use exact pixels
        WIDTH: int = Field(default=800, description="Width")
        HEIGHT: int = Field(default=1422, description="Height")
        # Aspect Ratio for models that use ratios
        ASPECT_RATIO: str = Field(
            default="16:9",
            description="Aspect Ratio. Options: 16:9, 1:1, 9:16, 4:3, 3:4",
        )
        # Model-Specific settings
        STYLE: str = Field(
            default="realistic_image", description="Style (Recraft only)"
        )
        RAW: bool = Field(default=False, description="Raw Mode (Flux Ultra only)")
        NUM_INFERENCE_STEPS: int = Field(
            default=28, description="Inference Steps (Hunyuan only)"
        )
        ENABLE_SAFETY_CHECKER: bool = Field(
            default=False, description="Enable Safety Checker"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "falai-master"
        self.name = "Fal.ai Master: "
        self.valves = self.Valves()
        self.emitter: Callable[[dict], Awaitable[None]] | None = None

    async def emit_status(self, message: str = "", done: bool = False):
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )

    def pipes(self) -> List[dict]:
        return [
            {"id": "falai-flux-ultra", "name": "IMG: Flux Ultra"},
            {"id": "falai-flux-2-pro", "name": "IMG: Flux 2 Pro"},
            {"id": "falai-recraft-v3", "name": "IMG: Recraft v3"},
            {"id": "falai-seedream", "name": "IMG: SeaDream"},
            {"id": "falai-hunyuan", "name": "IMG: Hunyuan"},
            {"id": "falai-imagen4", "name": "IMG: Imagen 4"},
            {"id": "falai-z-image", "name": "IMG: Z-Image Turbo"},
            {
                "id": "falai-custom",
                "name": "Fal.ai Custom (Use Valve)",
            },  # <--- ADDED THIS
        ]

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        self.emitter = __event_emitter__

        # 1. Determine Model ID
        request_model_id = body.get("model", "")

        # Map internal IDs to Fal API IDs
        model_map = {
            "falai-flux-ultra": "fal-ai/flux-pro/v1.1-ultra",
            "falai-flux-2-pro": "fal-ai/flux-2-pro",
            "falai-recraft-v3": "fal-ai/recraft/v3/text-to-image",
            "falai-seedream": "fal-ai/bytedance/seedream/v4.5/text-to-image",
            "falai-hunyuan": "fal-ai/hunyuan-image/v3/text-to-image",
            "falai-imagen4": "fal-ai/imagen4/preview/fast",
            "falai-z-image": "fal-ai/z-image/turbo",
            # Note: "falai-custom" is NOT here, so it falls through to the Valve
        }

        # If ID is in map, use it. If not (e.g. "falai-custom"), use the Valve value.
        api_model_id = model_map.get(request_model_id, self.valves.MODEL_ID)

        # 2. Get Prompt
        messages = body.get("messages", [])
        if not messages:
            return "Error: No messages found."

        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "") for p in content if p.get("type") == "text"
                    ]
                    prompt = " ".join(text_parts).strip()
                elif isinstance(content, str):
                    prompt = content
                break

        if not prompt:
            return "Error: No prompt found."

        # 3. Setup Env
        if not self.valves.FAL_KEY:
            return "Error: FAL_KEY not set in valves."
        os.environ["FAL_KEY"] = self.valves.FAL_KEY

        # 4. Construct Arguments
        arguments = {
            "prompt": prompt,
            "enable_safety_checker": self.valves.ENABLE_SAFETY_CHECKER,
        }

        # --- Helper for Recraft ---
        def get_recraft_size(ratio):
            mapping = {
                "1:1": "square_hd",
                "16:9": "landscape_16_9",
                "9:16": "portrait_16_9",
                "4:3": "landscape_4_3",
                "3:4": "portrait_4_3",
            }
            return mapping.get(ratio, "square_hd")

        # --- Logic Branches ---
        # We check if the api_model_id string contains known keywords

        if "recraft" in api_model_id:
            arguments["image_size"] = get_recraft_size(self.valves.ASPECT_RATIO)
            arguments["style"] = self.valves.STYLE

        elif "flux-pro/v1.1-ultra" in api_model_id:
            arguments["aspect_ratio"] = self.valves.ASPECT_RATIO
            arguments["raw"] = self.valves.RAW
            arguments["output_format"] = "jpeg"

        elif "imagen4" in api_model_id:
            arguments["aspect_ratio"] = self.valves.ASPECT_RATIO

        elif "hunyuan" in api_model_id:
            arguments["image_size"] = {
                "width": self.valves.WIDTH,
                "height": self.valves.HEIGHT,
            }
            arguments["num_inference_steps"] = self.valves.NUM_INFERENCE_STEPS

        elif "flux-2-pro" in api_model_id:
            arguments["width"] = self.valves.WIDTH
            arguments["height"] = self.valves.HEIGHT
            arguments["output_format"] = "jpeg"

        elif "seedream" in api_model_id or "z-image" in api_model_id:
            arguments["image_size"] = {
                "width": self.valves.WIDTH,
                "height": self.valves.HEIGHT,
            }

        else:
            # CUSTOM/Fallback Logic
            # If we are using a custom model from the Valve, we default to sending
            # the generic "image_size" object as it's the most common Fal pattern.
            arguments["image_size"] = {
                "width": self.valves.WIDTH,
                "height": self.valves.HEIGHT,
            }

        # 5. Call API
        await self.emit_status(f"🎨 Generating with {api_model_id}...")

        try:
            loop = asyncio.get_running_loop()

            def run_fal_generation():
                handler = fal_client.submit(
                    api_model_id,
                    arguments=arguments,
                )
                return handler.get()

            result = await loop.run_in_executor(None, run_fal_generation)

            if result and "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0].get("url", "")
                if image_url:
                    await self.emit_status("✅ Generated successfully", done=True)
                    return f"![Generated Image]({image_url})"
                else:
                    return "Error: Image URL missing."
            else:
                return f"Error: Generation failed. Result: {result}"

        except Exception as e:
            await self.emit_status(f"❌ Error: {str(e)}", done=True)
            return f"Error: {e}"
