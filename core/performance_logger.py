import csv
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging
import threading
from queue import Queue
from dataclasses import dataclass, asdict
import json
import pandas as pd
from enum import Enum, auto

class TradeAction(Enum):
    """Trade action types"""
    BUY = auto()
    SELL = auto()
    SHORT = auto()
    COVER = auto()
    DIVIDEND = auto()
    FEE = auto()
    ADJUSTMENT = auto()

class LogFormat(Enum):
    """Supported log formats"""
    CSV = auto()
    JSON = auto()
    PARQUET = auto()

@dataclass
class TradeRecord:
    """Structured trade record container"""
    timestamp: str
    symbol: str
    action: TradeAction
    price: float
    quantity: float
    pnl: Optional[float] = None
    commission: float = 0.0
    strategy: Optional[str] = None
    portfolio: Optional[str] = None
    risk_score: Optional[float] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class PerformanceLogger:
    """
    Professional-Grade Performance Logger with:
    - Multiple output formats (CSV, JSON, Parquet)
    - Thread-safe queued writing
    - Log rotation and compression
    - Comprehensive trade tracking
    - Performance analytics
    - Tagging and metadata support
    """
    
    def __init__(self,
                 filepath: str = "logs/performance",
                 log_format: LogFormat = LogFormat.CSV,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 7,
                 enable_compression: bool = True,
                 flush_interval: int = 60):  # seconds
        """
        Initialize performance logger
        
        Args:
            filepath: Base path for log files (extension will be added)
            log_format: Format for log files
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup logs to keep
            enable_compression: Whether to compress rotated logs
            flush_interval: How often to flush queue to disk (seconds)
        """
        self.log_format = log_format
        self.enable_compression = enable_compression
        self.base_path = Path(filepath)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Current log file handle
        self._current_file = None
        self._current_writer = None
        self._current_filename = None
        self._file_size = 0
        self._backup_count = backup_count
        self._max_file_size = max_file_size
        
        # Background queue for thread-safe logging
        self._log_queue = Queue(maxsize=10000)
        self._flush_event = threading.Event()
        self._shutdown_flag = False
        self._worker_thread = threading.Thread(
            target=self._process_log_queue,
            daemon=True,
            name="PerformanceLoggerWorker"
        )
        self._worker_thread.start()
        
        # Start periodic flusher
        self._flusher_thread = threading.Thread(
            target=self._periodic_flusher,
            daemon=True,
            kwargs={'interval': flush_interval}
        )
        self._flusher_thread.start()
        
        # Statistics
        self._stats = {
            'total_records': 0,
            'last_record_time': None,
            'queue_size': 0,
            'dropped_records': 0,
            'file_rotations': 0
        }
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize current log file
        self._rotate_file()
    
    def _get_file_extension(self) -> str:
        """Get file extension based on format"""
        extensions = {
            LogFormat.CSV: '.csv',
            LogFormat.JSON: '.json',
            LogFormat.PARQUET: '.parquet'
        }
        return extensions[self.log_format]
    
    def _get_current_filename(self) -> str:
        """Generate current log filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.base_path.stem}_{timestamp}{self._get_file_extension()}"
    
    def _rotate_file(self) -> None:
        """Rotate log file when size limit reached"""
        if self._current_file:
            self._current_file.close()
            if self.enable_compression:
                self._compress_file(self._current_filename)
        
        self._current_filename = self._get_current_filename()
        
        if self.log_format == LogFormat.CSV:
            self._current_file = open(self._current_filename, 'w', newline='')
            self._current_writer = csv.writer(self._current_file)
            # Write CSV header
            self._current_writer.writerow([
                'timestamp', 'symbol', 'action', 'price', 'quantity',
                'pnl', 'commission', 'strategy', 'portfolio', 'risk_score'
            ])
        elif self.log_format == LogFormat.JSON:
            self._current_file = open(self._current_filename, 'w')
        elif self.log_format == LogFormat.PARQUET:
            # We'll buffer records and write periodically for Parquet
            self._parquet_buffer = []
        
        self._file_size = 0
        with self._lock:
            self._stats['file_rotations'] += 1
    
    def _compress_file(self, filename: str) -> None:
        """Compress log file after rotation"""
        try:
            with open(filename, 'rb') as f_in:
                with gzip.open(f"{filename}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(filename)
        except Exception as e:
            self.logger.error(f"Error compressing log file {filename}: {e}")
    
    def _write_record(self, record: TradeRecord) -> None:
        """Write record to current log file"""
        record_dict = asdict(record)
        
        if self.log_format == LogFormat.CSV:
            self._current_writer.writerow([
                record_dict['timestamp'],
                record_dict['symbol'],
                record_dict['action'].name,
                record_dict['price'],
                record_dict['quantity'],
                record_dict['pnl'],
                record_dict['commission'],
                record_dict['strategy'],
                record_dict['portfolio'],
                record_dict['risk_score']
            ])
            self._file_size = self._current_file.tell()
        
        elif self.log_format == LogFormat.JSON:
            json.dump(record_dict, self._current_file)
            self._current_file.write('\n')
            self._file_size = self._current_file.tell()
        
        elif self.log_format == LogFormat.PARQUET:
            self._parquet_buffer.append(record_dict)
            if len(self._parquet_buffer) >= 1000:  # Write in batches
                self._write_parquet_buffer()
        
        # Rotate if file size exceeds limit
        if self._file_size > self._max_file_size:
            self._rotate_file()
    
    def _write_parquet_buffer(self) -> None:
        """Write buffered records to Parquet file"""
        if not self._parquet_buffer:
            return
            
        try:
            df = pd.DataFrame(self._parquet_buffer)
            # Convert enums to strings
            df['action'] = df['action'].apply(lambda x: x.name)
            
            if not os.path.exists(self._current_filename):
                df.to_parquet(self._current_filename)
            else:
                existing = pd.read_parquet(self._current_filename)
                combined = pd.concat([existing, df])
                combined.to_parquet(self._current_filename)
            
            self._parquet_buffer = []
            self._file_size = os.path.getsize(self._current_filename)
        except Exception as e:
            self.logger.error(f"Error writing Parquet file: {e}")
    
    def _process_log_queue(self) -> None:
        """Background thread to process log queue"""
        while not self._shutdown_flag:
            try:
                # Process all available records in the queue
                while not self._log_queue.empty():
                    record = self._log_queue.get()
                    self._write_record(record)
                    
                    with self._lock:
                        self._stats['total_records'] += 1
                        self._stats['last_record_time'] = datetime.now().isoformat()
                        self._stats['queue_size'] = self._log_queue.qsize()
                
                # Handle periodic flush
                if self._flush_event.wait(timeout=0.1):
                    self._flush_event.clear()
                    self._flush()
                
            except Exception as e:
                self.logger.error(f"Error processing log queue: {e}", exc_info=True)
    
    def _periodic_flusher(self, interval: int = 60) -> None:
        """Periodically flush records to disk"""
        while not self._shutdown_flag:
            time.sleep(interval)
            self._flush_event.set()
    
    def record(self, trade: Dict) -> None:
        """
        Record a trade with performance data
        
        Args:
            trade: Dictionary containing trade details
        """
        try:
            # Convert dict to TradeRecord
            record = TradeRecord(
                timestamp=trade.get('timestamp', datetime.utcnow().isoformat()),
                symbol=trade.get('symbol', ''),
                action=TradeAction[trade.get('action', 'BUY').upper()],
                price=float(trade.get('price', 0.0)),
                quantity=float(trade.get('quantity', 0.0)),
                pnl=float(trade.get('pnl', 0.0)) if trade.get('pnl') is not None else None,
                commission=float(trade.get('commission', 0.0)),
                strategy=trade.get('strategy'),
                portfolio=trade.get('portfolio'),
                risk_score=float(trade.get('risk_score')) if trade.get('risk_score') is not None else None,
                tags=trade.get('tags'),
                metadata=trade.get('metadata')
            )
            
            if not self._log_queue.full():
                self._log_queue.put(record)
            else:
                with self._lock:
                    self._stats['dropped_records'] += 1
                self.logger.warning("Performance log queue full - record dropped")
                
        except Exception as e:
            self.logger.error(f"Error processing trade record: {e}", exc_info=True)
    
    def _flush(self) -> None:
        """Flush all pending records to disk"""
        if self.log_format == LogFormat.PARQUET and self._parquet_buffer:
            self._write_parquet_buffer()
        
        if self._current_file:
            self._current_file.flush()
            os.fsync(self._current_file.fileno())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current logger statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['queue_size'] = self._log_queue.qsize()
        return stats
    
    def analyze_performance(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Analyze logged performance data
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            DataFrame with performance metrics
        """
        # Find relevant log files
        cutoff = datetime.now() - timedelta(days=lookback_days)
        log_files = []
        
        for f in self.base_path.parent.glob(f"{self.base_path.stem}_*{self._get_file_extension()}"):
            if f.stat().st_mtime >= cutoff.timestamp():
                log_files.append(f)
        
        # Read and combine log files
        records = []
        for f in sorted(log_files):
            try:
                if self.log_format == LogFormat.CSV:
                    df = pd.read_csv(f)
                    df['action'] = df['action'].apply(lambda x: TradeAction[x])
                    records.append(df)
                elif self.log_format == LogFormat.JSON:
                    with open(f) as jf:
                        records.extend([json.loads(line) for line in jf])
                elif self.log_format == LogFormat.PARQUET:
                    records.append(pd.read_parquet(f))
            except Exception as e:
                self.logger.error(f"Error reading log file {f}: {e}")
        
        if not records:
            return pd.DataFrame()
        
        if self.log_format in [LogFormat.CSV, LogFormat.PARQUET]:
            df = pd.concat(records)
        else:
            df = pd.DataFrame.from_records(records)
        
        return df
    
    def shutdown(self) -> None:
        """Clean shutdown of logger"""
        self._shutdown_flag = True
        self._flush_event.set()
        
        # Wait for threads to finish
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        if self._flusher_thread.is_alive():
            self._flusher_thread.join(timeout=1.0)
        
        # Final flush
        self._flush()
        if self._current_file:
            self._current_file.close()
        
        if self.log_format == LogFormat.PARQUET and self._parquet_buffer:
            self._write_parquet_buffer()
    
    def __del__(self):
        """Destructor - ensure proper cleanup"""
        self.shutdown()
