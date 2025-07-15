import dataclasses
import datetime
import importlib.util
import json
import logging
import os
import traceback

from opentelemetry import context as context_api
from opentelemetry.instrumentation.langchain.config import Config
from pydantic import BaseModel


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "AWS Open Telemetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper



def universal_debug_printer(**kwargs):
    """Universal debug printer that works with any LangChain callback handler.
    
    Usage: Just pass all arguments using **locals() at the start of any handler.
    Example: universal_debug_printer(**locals())
    """
    print("\n===== START DEBUG INFO =====")
    print(f"Callback: {kwargs.get('__name__', 'Unknown callback')}")
    print(f"Run ID: {kwargs.get('run_id', 'Unknown')}")
    print(f"Parent Run ID: {kwargs.get('parent_run_id', 'Unknown')}")
    
    # Handle serialized data (common to most callbacks)
    serialized = kwargs.get('serialized')
    if serialized:
        print("\n----- COMPONENT INFO -----")
        print(f"Name: {serialized.get('name', 'Unknown')}")
        print(f"ID: {serialized.get('id', 'Unknown')}")
        print(f"Type: {serialized.get('type', 'Unknown')}")
        
        # Print more detailed serialized info but handle potential large data
        print("\nDetailed Component Data:")
        try:
            print(json.dumps(serialized, indent=2, default=str))
        except Exception as e:
            print(f"Could not JSON serialize full object: {e}")
            print(f"Available keys: {list(serialized.keys())}")
    
    # Handle different types of inputs
    if 'inputs' in kwargs:
        print("\n----- INPUTS -----")
        inputs = kwargs['inputs']
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                print(f"{k}: {truncate_text(v)}")
        else:
            print(truncate_text(inputs))
    
    # Handle prompts (for LLM callbacks)
    if 'prompts' in kwargs:
        print("\n----- PROMPTS -----")
        for i, prompt in enumerate(kwargs['prompts']):
            print(f"Prompt {i}: {truncate_text(prompt)}")
    
    # Handle messages (for chat model callbacks)
    if 'messages' in kwargs:
        print("\n----- MESSAGES -----")
        for i, msg_list in enumerate(kwargs['messages']):
            print(f"Message set {i}:")
            for j, msg in enumerate(msg_list):
                print(f"  {j}. {msg.type}: {truncate_text(msg.content)}")
    
    # Handle tool inputs
    if 'input_str' in kwargs:
        print("\n----- TOOL INPUT -----")
        print(truncate_text(kwargs['input_str']))
    
    # Print tags
    tags = kwargs.get('tags')
    if tags:
        print("\n----- TAGS -----")
        print(tags)
    
    # Print metadata
    metadata = kwargs.get('metadata')
    if metadata:
        print("\n----- METADATA -----")
        try:
            print(json.dumps(metadata, indent=2, default=str))
        except Exception as e:
            print(f"Could not JSON serialize metadata: {e}")
            if metadata:
                print(f"Metadata keys: {list(metadata.keys())}")
    
    # Print other kwargs (excluding the ones we've already processed)
    other_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['serialized', 'inputs', 'prompts', 'messages', 
                               'input_str', 'run_id', 'parent_run_id', 
                               'tags', 'metadata', '__name__']}
    if other_kwargs:
        print("\n----- ADDITIONAL PARAMETERS -----")
        for key, value in other_kwargs.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {type(value)} (complex object)")
    
    print("\n===== END DEBUG INFO =====")

def truncate_text(text, max_length=200):
    """Truncate text to a maximum length."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_length:
        return f"{text[:max_length]}... (truncated, total length: {len(text)})"
    return text