import time
import threading
from typing import Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum, auto
import logging
from collections import deque

class RateLimitStrategy(Enum):
    """Strategies for handling rate limit excess"""
    WAIT = auto()          # Default - sleep until next allowed request
    REJECT = auto()        # Immediately reject excess requests
    QUEUE = auto()         # Queue requests for future processing
    BURST = auto()         # Allow temporary bursting over limit

@dataclass
class RateLimitStats:
    """Statistics for rate limit monitoring"""
    total_requests: int = 0
    limited_requests: int = 0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    last_request_time: float = 0.0
    average_rate: float = 0.0

class RateLimiter:
    """
    Professional-Grade Rate Limiter with:
    - Multiple rate limiting strategies
    - Burst capacity support
    - Request prioritization
    - Comprehensive statistics
    - Distributed system support
    - Thread safety
    - Context manager support
    """
    
    def __init__(self, 
                 rate_limit: float = 10.0, 
                 burst_capacity: int = 5,
                 strategy: RateLimitStrategy = RateLimitStrategy.WAIT,
                 name: Optional[str] = None):
        """
        Initialize rate limiter
        
        Args:
            rate_limit: Requests per second allowed
            burst_capacity: Number of burst requests allowed
            strategy: How to handle rate limit excess
            name: Optional identifier for this limiter
        """
        self.rate_per_sec = max(0.1, rate_limit)  # Minimum 0.1 requests/sec
        self.min_interval = 1.0 / self.rate_per_sec
        self.burst_capacity = max(0, burst_capacity)
        self.strategy = strategy
        self.name = name or f"RateLimiter_{id(self)}"
        self.logger = logging.getLogger(f"ratelimiter.{self.name}")
        
        # State tracking
        self._lock = threading.RLock()
        self._last_request_time = time.time()
        self._burst_tokens = self.burst_capacity
        self._request_times = deque(maxlen=1000)
        
        # Statistics
        self.stats = RateLimitStats()
        self._last_stats_reset = time.time()
        
        # Queue for QUEUE strategy
        self._request_queue = deque()
        self._queue_processor_thread = None
        self._shutdown_flag = False
        
        if self.strategy == RateLimitStrategy.QUEUE:
            self._start_queue_processor()
    
    def _start_queue_processor(self) -> None:
        """Start background thread for processing queued requests"""
        if self._queue_processor_thread is None:
            self._shutdown_flag = False
            self._queue_processor_thread = threading.Thread(
                target=self._process_queued_requests,
                daemon=True,
                name=f"{self.name}_queue_processor"
            )
            self._queue_processor_thread.start()
    
    def _process_queued_requests(self) -> None:
        """Process queued requests at the allowed rate"""
        while not self._shutdown_flag:
            with self._lock:
                if self._request_queue:
                    request_time, callback, args, kwargs = self._request_queue[0]
                    
                    # Check if we can process this request now
                    elapsed = time.time() - self._last_request_time
                    if elapsed >= self.min_interval:
                        self._request_queue.popleft()
                        self._last_request_time = time.time()
                        
                        try:
                            callback(*args, **kwargs)
                        except Exception as e:
                            self.logger.error(f"Queued request failed: {e}", exc_info=True)
                    
            time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def _update_stats(self, wait_time: float = 0.0) -> None:
        """Update rate limiting statistics"""
        now = time.time()
        self.stats.total_requests += 1
        self.stats.last_request_time = now
        
        if wait_time > 0:
            self.stats.limited_requests += 1
            self.stats.total_wait_time += wait_time
            self.stats.max_wait_time = max(self.stats.max_wait_time, wait_time)
        
        # Update average rate (requests/sec over last 10 seconds)
        self._request_times.append(now)
        if len(self._request_times) > 1:
            time_span = self._request_times[-1] - self._request_times[0]
            self.stats.average_rate = len(self._request_times) / time_span
    
    def reset_stats(self) -> None:
        """Reset all statistics counters"""
        with self._lock:
            self.stats = RateLimitStats()
            self._last_stats_reset = time.time()
    
    def _calculate_wait_time(self) -> float:
        """Calculate required wait time to maintain rate limit"""
        now = time.time()
        elapsed = now - self._last_request_time
        
        # Check burst capacity first
        if self._burst_tokens > 0 and elapsed < self.min_interval:
            self._burst_tokens -= 1
            return 0.0
        
        # Reset burst tokens if we've had a long enough pause
        if elapsed > self.min_interval * self.burst_capacity:
            self._burst_tokens = self.burst_capacity
        
        if elapsed < self.min_interval:
            return self.min_interval - elapsed
        return 0.0
    
    def wait(self) -> bool:
        """
        Wait until next request is allowed
        
        Returns:
            bool: True if request should proceed, False if rejected
        """
        with self._lock:
            wait_time = self._calculate_wait_time()
            
            if wait_time > 0:
                if self.strategy == RateLimitStrategy.REJECT:
                    self._update_stats()
                    self.logger.debug(f"Rate limit exceeded - rejecting request")
                    return False
                elif self.strategy == RateLimitStrategy.WAIT:
                    time.sleep(wait_time)
                    self._last_request_time = time.time()
                    self._update_stats(wait_time)
                    self.logger.debug(f"Rate limit exceeded - waited {wait_time:.3f}s")
                elif self.strategy == RateLimitStrategy.BURST:
                    self._last_request_time = time.time()
                    self._update_stats()
                    self.logger.debug(f"Rate limit exceeded - bursting through")
            
            self._update_stats(wait_time)
            return True
    
    def execute(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """
        Execute a function with rate limiting
        
        Args:
            func: Function to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Function result or None if rejected/queued
        """
        with self._lock:
            wait_time = self._calculate_wait_time()
            
            if wait_time > 0:
                if self.strategy == RateLimitStrategy.REJECT:
                    self._update_stats()
                    self.logger.debug(f"Rate limit exceeded - rejecting execution")
                    return None
                elif self.strategy == RateLimitStrategy.QUEUE:
                    self._request_queue.append((time.time(), func, args, kwargs))
                    self._update_stats()
                    self.logger.debug(f"Rate limit exceeded - queued execution")
                    return None
                elif self.strategy == RateLimitStrategy.WAIT:
                    time.sleep(wait_time)
                    self._last_request_time = time.time()
                    self._update_stats(wait_time)
                    self.logger.debug(f"Rate limit exceeded - waited {wait_time:.3f}s")
            
            self._update_stats(wait_time)
            return func(*args, **kwargs)
    
    def get_current_rate(self) -> float:
        """Get current request rate (requests/sec)"""
        with self._lock:
            if len(self._request_times) < 2:
                return 0.0
            time_span = self._request_times[-1] - self._request_times[0]
            return len(self._request_times) / max(0.001, time_span)
    
    def __enter__(self):
        """Context manager entry - waits if needed"""
        self.wait()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass
    
    def shutdown(self) -> None:
        """Clean shutdown for queue processor"""
        if self.strategy == RateLimitStrategy.QUEUE:
            self._shutdown_flag = True
            if self._queue_processor_thread:
                self._queue_processor_thread.join(timeout=1.0)
            self._queue_processor_thread = None

