from typing import Any, List, Optional
from smolagents.models import LiteLLMModel
from smolagents.models import ChatMessage, ChatMessageStreamDelta

class SafeLiteLLMModel(LiteLLMModel):
    """
    A subclass of LiteLLMModel that intercepts stop_sequences.
    This prevents the underlying backend (like Transformers Server) 
    from crashing due to its inability to handle stop_sequences implicitly,
    truncating the generated text correctly on the client side instead.
    """

    def generate(self, messages: Any, stop_sequences: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> ChatMessage:
        # Call the underlying model without stop sequences
        response: ChatMessage = super().generate(messages, None, *args, **kwargs)
        
        # Client-side truncation of the stop_sequences
        if stop_sequences and response.content:
            min_index = len(response.content)
            truncated = False
            for stop_seq in stop_sequences:
                index = response.content.find(stop_seq)
                if index != -1 and index < min_index:
                    min_index = index
                    truncated = True
            
            if truncated:
                response.content = response.content[:min_index]
                    
        return response

    def generate_stream(self, messages: Any, stop_sequences: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> Any:
        buffer = ""
        stop_seqs = stop_sequences or []
        for delta in super().generate_stream(messages, None, *args, **kwargs):
            if type(delta.content) is str:
                # append delta content to buffer
                buffer += delta.content
                
                # Check if buffer contains any stop sequence
                matched_stop = None
                for stop_seq in stop_seqs:
                    if stop_seq in buffer:
                        matched_stop = stop_seq
                        break
                
                if matched_stop is not None:
                    # Truncate buffer at stop sequence
                    idx = buffer.find(matched_stop)
                    valid_text = buffer[:idx]
                    
                    # Yield remaining valid text
                    delta.content = valid_text
                    yield delta
                    return  # break the generator
                
                # Suffix check to prevent yielding prefix of a stop sequence
                max_suffix_len = 0
                for stop_seq in stop_seqs:
                    # check overlapping suffixes up to len(stop_seq) - 1
                    for i in range(1, len(stop_seq)):
                        if buffer.endswith(stop_seq[:i]):
                            max_suffix_len = max(max_suffix_len, i)
                            
                if max_suffix_len == 0:
                    safe_text = buffer
                    buffer = ""
                else:
                    safe_text = buffer[:-max_suffix_len]
                    buffer = buffer[-max_suffix_len:]
                
                if safe_text:
                    delta.content = safe_text
                    yield delta
                else:
                    # Yield an empty delta to keep the stream moving for UI without leaking potential stop characters
                    delta.content = ""
                    yield delta
            else:
                yield delta
                
        # Flush any remaining non-stop text
        if buffer:
            yield ChatMessageStreamDelta(content=buffer)
