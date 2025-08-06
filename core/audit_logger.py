import logging
from datetime import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
import gzip
import threading
from queue import Queue
from enum import Enum, auto
from dataclasses import dataclass, asdict
import uuid

class LogLevel(Enum):
    """Standard log levels with additional audit-specific levels"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    AUDIT = auto()  # Special level for audit trails
    SECURITY = auto()  # Security-related events

@dataclass
class AuditEvent:
    """Structured audit event container"""
    event_id: str
    timestamp: str
    action: str
    actor: Optional[str] = None
    status: str = "SUCCESS"
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    correlation_id: Optional[str] = None

class AuditLogger:
    """
    Professional-Grade Audit Logger with:
    - Structured JSON logging
    - Thread-safe queued writing
    - Log rotation and compression
    - Multiple output handlers
    - Event correlation
    - Security event tracking
    - Performance metrics
    """
    
    def __init__(self, 
                 log_path: Union[str, Path] = "logs/audit.log",
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 7,
                 enable_console: bool = False,
                 enable_syslog: bool = False,
                 syslog_address: Optional[tuple[str, int]] = None):
        """
        Initialize audit logger
        
        Args:
            log_path: Path to audit log file
            max_file_size: Maximum log file size before rotation (bytes)
            backup_count: Number of backup logs to keep
            enable_console: Mirror logs to console
            enable_syslog: Enable syslog forwarding
            syslog_address: (host, port) for syslog server
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
        
        # Optional console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)
        
        # Optional syslog handler
        if enable_syslog and syslog_address:
            syslog_handler = logging.handlers.SysLogHandler(
                address=syslog_address,
                facility=logging.handlers.SysLogHandler.LOG_LOCAL0
            )
            self.logger.addHandler(syslog_handler)
        
        # Background queue for thread-safe logging
        self._log_queue = Queue(maxsize=10000)
        self._worker_thread = threading.Thread(
            target=self._process_log_queue,
            daemon=True,
            name="AuditLoggerWorker"
        )
        self._worker_thread.start()
        
        # Statistics
        self._stats = {
            'total_events': 0,
            'events_by_level': {level.name: 0 for level in LogLevel},
            'last_event_time': None,
            'queue_size': 0,
            'dropped_events': 0
        }
        self._lock = threading.Lock()
    
    def _process_log_queue(self) -> None:
        """Background thread to process log queue"""
        while True:
            try:
                event = self._log_queue.get()
                if isinstance(event, AuditEvent):
                    self._log_structured_event(event)
                else:
                    self.logger.log(event['level'], event['message'])
                
                with self._lock:
                    self._stats['queue_size'] = self._log_queue.qsize()
            except Exception as e:
                logging.error(f"Error processing log queue: {e}", exc_info=True)
    
    def _log_structured_event(self, event: AuditEvent) -> None:
        """Log structured audit event"""
        try:
            log_entry = json.dumps(asdict(event), ensure_ascii=False)
            self.logger.info(log_entry)
            
            with self._lock:
                self._stats['total_events'] += 1
                self._stats['events_by_level'][LogLevel.AUDIT.name] += 1
                self._stats['last_event_time'] = datetime.utcnow().isoformat()
        except Exception as e:
            logging.error(f"Error logging audit event: {e}", exc_info=True)
    
    def log_event(self,
                 action: str,
                 status: str = "SUCCESS",
                 actor: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None,
                 resource: Optional[str] = None,
                 correlation_id: Optional[str] = None) -> str:
        """
        Log a structured audit event
        
        Args:
            action: Description of the action being logged
            status: SUCCESS/FAILURE/ATTEMPT
            actor: Who performed the action
            details: Additional event details
            ip_address: Source IP address
            user_agent: User agent string
            resource: Resource being acted upon
            correlation_id: Correlation ID for tracing
            
        Returns:
            Generated event ID
        """
        event_id = str(uuid.uuid4())
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            actor=actor,
            status=status,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            correlation_id=correlation_id or self._get_current_correlation_id()
        )
        
        try:
            if not self._log_queue.full():
                self._log_queue.put(event)
            else:
                with self._lock:
                    self._stats['dropped_events'] += 1
                logging.warning("Audit log queue full - event dropped")
        except Exception as e:
            logging.error(f"Error queueing audit event: {e}", exc_info=True)
        
        return event_id
    
    def log_security_event(self,
                         action: str,
                         threat_level: str = "MEDIUM",
                         **kwargs) -> str:
        """
        Log a security-related event with additional metadata
        
        Args:
            action: Description of the security event
            threat_level: LOW/MEDIUM/HIGH/CRITICAL
            **kwargs: Additional AuditEvent fields
            
        Returns:
            Generated event ID
        """
        details = kwargs.get('details', {})
        details['threat_level'] = threat_level
        kwargs['details'] = details
        
        event_id = self.log_event(action, **kwargs)
        
        with self._lock:
            self._stats['events_by_level'][LogLevel.SECURITY.name] += 1
        
        # Also log at critical level for immediate attention
        self._log_queue.put({
            'level': logging.CRITICAL if threat_level in ["HIGH", "CRITICAL"] else logging.WARNING,
            'message': f"SECURITY EVENT ({threat_level}): {action}"
        })
        
        return event_id
    
    def _get_current_correlation_id(self) -> Optional[str]:
        """Get current correlation ID from context (e.g., Flask/W3C Trace Context)"""
        # In a real implementation, this would get the ID from request context
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current logger statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['queue_size'] = self._log_queue.qsize()
        return stats
    
    def compress_logs(self) -> None:
        """Compress rotated log files"""
        log_dir = self.log_path.parent
        for file in log_dir.glob(f"{self.log_path.name}.*"):
            if not file.name.endswith('.gz') and file.exists():
                try:
                    with open(file, 'rb') as f_in:
                        with gzip.open(f"{file}.gz", 'wb') as f_out:
                            f_out.writelines(f_in)
                    file.unlink()
                except Exception as e:
                    logging.error(f"Error compressing log {file}: {e}")
    
    def flush(self) -> None:
        """Flush all pending log events"""
        while not self._log_queue.empty():
            time.sleep(0.1)  # Allow queue to drain
    
    def __del__(self):
        """Cleanup on instance destruction"""
        self.flush()
