"""
title: Fal.ai Z-Image Turbo Pipe
description: A pipe to generate images using fal.ai's z-image/turbo model.
author: User
version: 1.0.0
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
        FAL_KEY: str = Field(
            default="",
            description="API Key for Fal.ai (required)"
        )
        WIDTH: int = Field(
            default=832,
            description="Image width"
        )
        HEIGHT: int = Field(
            default=1216,
            description="Image height"
        )
        ENABLE_SAFETY_CHECKER: bool = Field(
            default=False,
            description="Enable safety checker"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "falai-z-image-turbo"
        self.name = "Fal.ai: "
        self.valves = self.Valves()
        self.emitter: Callable[[dict], Awaitable[None]] | None = None

    async def emit_status(self, message: str = "", done: bool = False):
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )

    def pipes(self) -> List[dict]:
        return [{"id": "falai-z-image-turbo", "name": "Fal.ai Z-Image Turbo"}]

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        self.emitter = __event_emitter__
        
        # 1. Get the prompt from the last user message
        messages = body.get("messages", [])
        if not messages:
            return "Error: No messages found."
            
        # Extract the last user message content
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    # Handle multimodal content (text + images), extract text parts
                    text_parts = []
                    for part in content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    prompt = " ".join(text_parts).strip()
                elif isinstance(content, str):
                    prompt = content
                break
        
        if not prompt:
             return "Error: No prompt found in the last user message."

        # 2. Setup Configuration from Valves
        if not self.valves.FAL_KEY:
            return "Error: FAL_KEY not set in valves. Please configure it in the function settings."

        # Set the API key for fal_client (fal_client checks os.environ or takes it directly, 
        # but setting os.environ is the safest way to ensure the library picks it up globally)
        os.environ["FAL_KEY"] = self.valves.FAL_KEY
        
        # 3. Call Fal.ai API
        await self.emit_status("🎨 Generating image with Fal.ai...")
        
        try:
            # We run the blocking fal_client.submit in a separate thread to not block the async loop
            loop = asyncio.get_running_loop()
            
            def run_fal_generation():
                handler = fal_client.submit(
                    "fal-ai/z-image/turbo",
                    arguments={
                        "prompt": prompt,
                        "enable_safety_checker": self.valves.ENABLE_SAFETY_CHECKER,
                        "image_size": {
                            "width": self.valves.WIDTH,
                            "height": self.valves.HEIGHT
                        }
                    },
                )
                return handler.get()

            result = await loop.run_in_executor(None, run_fal_generation)
            
            # 4. Process Result
            if result and 'images' in result and len(result['images']) > 0:
                image_url = result['images'][0].get('url', '')
                if image_url:
                    await self.emit_status("✅ Image generated successfully", done=True)
                    # Return markdown image syntax
                    return f"![Generated Image]({image_url})"
                else:
                    await self.emit_status("❌ Failed to retrieve image URL", done=True)
                    return "Error: Image generated but URL was missing."
            else:
                 await self.emit_status("❌ Generation failed or empty result", done=True)
                 return f"Error: Generation failed. Result: {result}"

        except Exception as e:
            await self.emit_status(f"❌ Error: {str(e)}", done=True)
            return f"Error generating image: {e}"
