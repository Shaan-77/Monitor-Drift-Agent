"""
Terminal error capture module.

This module automatically captures all errors from stderr (terminal)
and stores them in a .log file. It filters for actual errors only,
avoiding false positives.
"""

import sys
import os
from datetime import datetime
from typing import TextIO
import re


class ErrorCapture:
    """
    Captures errors from stderr and writes them to a log file.
    
    Only captures actual errors (exceptions, error messages, tracebacks)
    to avoid false positives.
    """
    
    # Patterns that indicate actual errors
    # These patterns are designed to match actual errors, not just messages about errors
    ERROR_PATTERNS = [
        r'Traceback\s+\(most recent call last\)',  # Python traceback start
        r'File\s+"[^"]+",\s+line\s+\d+',  # Traceback file reference
        r'^\w+Error:',  # Exception type at start of line (e.g., "ValueError:", "TypeError:")
        r'^\w+Exception:',  # Exception type at start of line
        r'^\s+\^',  # Python 3.11+ error indicator
        r'^\s+~\^',  # Python 3.11+ error indicator with tildes
    ]
    
    # Additional patterns that indicate errors when found in context
    ERROR_CONTEXT_PATTERNS = [
        r'\bERROR\b',  # ERROR as a word (not part of another word)
        r'\bCRITICAL\b',  # CRITICAL as a word
        r'\bFatal\s+error\b',  # Fatal error
        r'\bFatal\s+exception\b',  # Fatal exception
    ]
    
    def __init__(self, log_file: str = "errors.log"):
        """
        Initialize error capture.
        
        Args:
            log_file: Path to the log file where errors will be stored
        """
        self.log_file = log_file
        self.original_stderr = sys.stderr
        self.error_buffer = []
        self.buffer_size = 100  # Buffer lines to detect multi-line errors
        self.is_capturing = False
        self.in_traceback = False  # Track if we're currently in a traceback
        
        # Compile error patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ERROR_PATTERNS]
        self.context_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ERROR_CONTEXT_PATTERNS]
        
        # Pattern to detect traceback start
        self.traceback_start_pattern = re.compile(r'Traceback\s+\(most recent call last\)', re.IGNORECASE)
    
    def _is_error_line(self, line: str, check_context: bool = True) -> bool:
        """
        Check if a line contains an error pattern.
        
        Uses strict matching to avoid false positives:
        - Primary patterns match actual error structures (tracebacks, exceptions)
        - Context patterns only match when in error context (traceback, exception nearby)
        
        Args:
            line: Line of text to check
            check_context: Whether to check context patterns (default: True)
            
        Returns:
            True if line contains an error pattern, False otherwise
        """
        if not line or not line.strip():
            return False
        
        # Check against primary error patterns (always indicate errors)
        for pattern in self.compiled_patterns:
            if pattern.search(line):
                return True
        
        # Check context patterns only if we're in an error context
        # This prevents false positives from informational messages
        if check_context and (self.in_traceback or self._has_recent_error_context()):
            for pattern in self.context_patterns:
                if pattern.search(line):
                    return True
        
        return False
    
    def _has_recent_error_context(self) -> bool:
        """
        Check if recent buffer contains error context.
        
        Returns:
            True if recent buffer lines contain primary error patterns
        """
        # Check last few buffer lines for primary error patterns
        for buffered_line in self.error_buffer[-5:]:
            for pattern in self.compiled_patterns:
                if pattern.search(buffered_line):
                    return True
        return False
    
    def _write_to_log(self, content: str):
        """
        Write content to the error log file.
        
        Args:
            content: Content to write to the log file
        """
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Append to log file with timestamp
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {content}\n")
        except Exception as e:
            # Fallback to original stderr if log writing fails
            self.original_stderr.write(f"Error writing to log file: {e}\n")
            self.original_stderr.write(content)
    
    def _process_line(self, line: str):
        """
        Process a line from stderr.
        
        Args:
            line: Line from stderr
        """
        # Check if this line starts a traceback
        if self.traceback_start_pattern.search(line):
            self.in_traceback = True
        
        # Always write to original stderr first so user sees output immediately
        self.original_stderr.write(line)
        self.original_stderr.flush()
        
        # Add to buffer for error detection
        self.error_buffer.append(line)
        
        # Keep buffer size manageable
        if len(self.error_buffer) > self.buffer_size:
            self.error_buffer.pop(0)
        
        # Check if this line contains an error pattern
        is_error = self._is_error_line(line)
        
        # If we're in a traceback, continue capturing until we see an exception
        if self.in_traceback:
            # Check if this line contains an exception (end of traceback)
            if re.search(r'^\w+Error:|^\w+Exception:', line, re.IGNORECASE):
                self.in_traceback = False
                is_error = True
                # Write complete traceback to log
                error_context = ''.join(self.error_buffer)
                self._write_to_log(error_context)
                # Clear buffer after logging
                self.error_buffer = []
        else:
            # Also check recent buffer for context (tracebacks span multiple lines)
            if not is_error and len(self.error_buffer) > 1:
                # Check last few lines together for traceback patterns
                recent_context = ''.join(self.error_buffer[-10:])
                is_error = self._is_error_line(recent_context)
            
            if is_error:
                # Write the error to log
                error_context = ''.join(self.error_buffer)
                self._write_to_log(error_context)
                # Clear buffer after writing
                self.error_buffer = []
    
    def start_capture(self):
        """Start capturing errors from stderr."""
        if self.is_capturing:
            return
        
        # Replace stderr with our wrapper
        sys.stderr = self
        self.is_capturing = True
        
        # Write initialization message
        init_msg = f"Error capture started. Errors will be logged to: {os.path.abspath(self.log_file)}\n"
        self.original_stderr.write(init_msg)
        self.original_stderr.flush()
    
    def stop_capture(self):
        """Stop capturing errors and restore original stderr."""
        if not self.is_capturing:
            return
        
        # Flush any remaining buffer
        if self.error_buffer:
            # Check if buffer contains errors before writing
            buffer_content = ''.join(self.error_buffer)
            if self._is_error_line(buffer_content):
                self._write_to_log(buffer_content)
            else:
                # Write non-error content to original stderr
                self.original_stderr.write(buffer_content)
        
        # Restore original stderr
        sys.stderr = self.original_stderr
        self.is_capturing = False
        self.error_buffer = []
    
    def write(self, text: str):
        """
        Implement file-like interface for stderr replacement.
        
        This method is called when code writes to sys.stderr.
        
        Args:
            text: Text to write to stderr
        """
        if not text:
            return
        
        # Split by newlines to process line by line
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                # Not the last line, add newline
                self._process_line(line + '\n')
            else:
                # Last line, process as-is (may not have newline)
                if line:
                    self._process_line(line)
    
    def flush(self):
        """Implement file-like interface for stderr replacement."""
        self.original_stderr.flush()
    
    def writable(self):
        """Implement file-like interface - stderr is always writable."""
        return True
    
    def isatty(self):
        """Implement file-like interface - check if original stderr is a TTY."""
        return self.original_stderr.isatty() if hasattr(self.original_stderr, 'isatty') else False
    
    def fileno(self):
        """Implement file-like interface - return file descriptor."""
        return self.original_stderr.fileno() if hasattr(self.original_stderr, 'fileno') else -1
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()
        return False


# Global error capture instance
_error_capture_instance = None


def initialize_error_capture(log_file: str = "errors.log"):
    """
    Initialize global error capture.
    
    This function should be called early in the application startup
    to begin capturing errors automatically.
    
    Args:
        log_file: Path to the log file (default: "errors.log")
    """
    global _error_capture_instance
    
    if _error_capture_instance is None:
        _error_capture_instance = ErrorCapture(log_file)
        _error_capture_instance.start_capture()
    
    return _error_capture_instance


def get_error_capture() -> ErrorCapture:
    """
    Get the global error capture instance.
    
    Returns:
        ErrorCapture instance or None if not initialized
    """
    return _error_capture_instance


def stop_error_capture():
    """Stop the global error capture."""
    global _error_capture_instance
    
    if _error_capture_instance:
        _error_capture_instance.stop_capture()
        _error_capture_instance = None
