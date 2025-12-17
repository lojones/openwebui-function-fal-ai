"""
title: Fal.ai Master Image Generator
description: A unified pipe to generate images using various Fal.ai models. Requires explicit model selection.
author: User
version: 2.2.0
license: MIT
requirements: fal-client, python-dotenv
environment_variables: FAL_KEY
"""

import os
import asyncio
from typing import List, Callable, Awaitable, AsyncGenerator
from pydantic import BaseModel, Field
import fal_client


class Pipe:
    class Valves(BaseModel):
        FAL_KEY: str = Field(default="", description="API Key for Fal.ai (required)")
        # Dimensions for models that use exact pixels (Flux 2, Hunyuan, SeaDream, Z-Image)
        WIDTH: int = Field(default=800, description="Width")
        HEIGHT: int = Field(default=1422, description="Height")
        # Aspect Ratio for models that use ratios (Flux Ultra, Imagen 4, Recraft)
        ASPECT_RATIO: str = Field(
            default="9:16",
            description="Aspect Ratio. Options: 16:9, 1:1, 9:16, 4:3, 3:4",
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
            try:
                await self.emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": message,
                            "done": done,
                        },
                    }
                )
            except Exception:
                pass

    def pipes(self) -> List[dict]:
        return [
            {"id": "falai-flux-2-pro", "name": "IMG: Flux 2 Pro"},
            {"id": "falai-recraft-v3", "name": "IMG: Recraft v3"},
            {"id": "falai-seedream", "name": "IMG: SeaDream"},
            {"id": "falai-hunyuan", "name": "IMG: Hunyuan"},
            {"id": "falai-imagen4", "name": "IMG: Imagen 4"},
            {"id": "falai-z-image", "name": "IMG: Z-Image Turbo"},
        ]

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> AsyncGenerator[str, None]:
        self.emitter = __event_emitter__

        # 1. Determine Model ID
        request_model_id = body.get("model", "")

        # Map internal IDs to Fal API IDs
        model_map = {
            "falai-flux-2-pro": "fal-ai/flux-2-pro",
            "falai-recraft-v3": "fal-ai/recraft/v3/text-to-image",
            "falai-seedream": "fal-ai/bytedance/seedream/v4.5/text-to-image",
            "falai-hunyuan": "fal-ai/hunyuan-image/v3/text-to-image",
            "falai-imagen4": "fal-ai/imagen4/preview/fast",
            "falai-z-image": "fal-ai/z-image/turbo",
        }

        api_model_id = None

        # Check for known models
        for internal_id, external_id in model_map.items():
            if internal_id in request_model_id:
                api_model_id = external_id
                break

        # If no match found, return ERROR immediately
        if not api_model_id:
            yield f"**Error:** The selected model (`{request_model_id}`) is not supported by the Fal.ai Master Pipe.\n\nPlease select one of the **IMG:** models from the dropdown list."
            return

        # 2. Get Prompt
        messages = body.get("messages", [])
        if not messages:
            yield "Error: No messages found."
            return

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
            yield "Error: No prompt found."
            return

        # 3. Setup Env
        if not self.valves.FAL_KEY:
            yield "Error: FAL_KEY not set in valves."
            return
        os.environ["FAL_KEY"] = self.valves.FAL_KEY

        # 4. Construct Arguments
        arguments = {
            "prompt": prompt,
            "enable_safety_checker": self.valves.ENABLE_SAFETY_CHECKER,
        }

        # Logic Branches based on selected model

        if "imagen4" in api_model_id:
            arguments["aspect_ratio"] = self.valves.ASPECT_RATIO
        else: 
            # Fallback for known models that might not have a specific block above
            arguments["image_size"] = {
                "width": self.valves.WIDTH,
                "height": self.valves.HEIGHT,
            }

        # 5. Call API
        await self.emit_status(f"Generating image with {api_model_id}...", done=False)

        try:
            loop = asyncio.get_running_loop()

            def run_fal_generation():
                return fal_client.submit(api_model_id, arguments=arguments).get()

            result = await loop.run_in_executor(None, run_fal_generation)

            if result and "images" in result and len(result["images"]) > 0:
                image_url = result["images"][0].get("url", "")
                if image_url:
                    await self.emit_status("Generation successful", done=True)
                    yield f"![Generated Image]({image_url})"
                else:
                    await self.emit_status("Error: Image URL missing", done=True)
                    yield "Error: Image URL missing."
            else:
                await self.emit_status("Generation failed", done=True)
                yield f"Error: Generation failed. Result: {result}"

        except Exception as e:
            await self.emit_status(f"Error: {e}", done=True)
            yield f"Error: {e}"
