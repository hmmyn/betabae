#!/usr/bin/env python3
"""
MicroBetaBae Scalability Monitor
Monitor the 1 million episode training run for performance and memory usage
"""

import time
import psutil
import os
import json
from pathlib import Path
import subprocess

def get_process_info(pid):
    """Get process information"""
    try:
        process = psutil.Process(pid)
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'num_threads': process.num_threads(),
            'status': process.status()
        }
    except psutil.NoSuchProcess:
        return None

def monitor_training():
    """Monitor the training process"""
    print("MicroBetaBae Scalability Monitor")
    print("=" * 50)
    
    # Find the training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'micro_train' in ' '.join(proc.info['cmdline'] or []):
                training_pid = proc.info['pid']
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not training_pid:
        print("❌ Training process not found!")
        return
    
    print(f"✅ Found training process: PID {training_pid}")
    
    # Monitor for 60 seconds
    start_time = time.time()
    episode_count = 0
    last_check = start_time
    
    while time.time() - start_time < 60:  # Monitor for 1 minute
        # Get process info
        proc_info = get_process_info(training_pid)
        if not proc_info:
            print("❌ Training process ended!")
            break
        
        # Check for new episode files
        output_dir = Path('./micro_million')
        if output_dir.exists():
            stats_files = list(output_dir.glob('stats_ep_*.json'))
            if stats_files:
                latest_file = max(stats_files, key=lambda x: int(x.stem.split('_')[-1]))
                with open(latest_file, 'r') as f:
                    stats = json.load(f)
                episode_count = len(stats['rewards'])
        
        # Calculate performance metrics
        current_time = time.time()
        elapsed = current_time - start_time
        episodes_per_sec = episode_count / elapsed if elapsed > 0 else 0
        
        # Display status
        print(f"\r🔄 Episode {episode_count:,} | "
              f"Speed: {episodes_per_sec:.1f} ep/s | "
              f"CPU: {proc_info['cpu_percent']:.1f}% | "
              f"Memory: {proc_info['memory_mb']:.1f} MB | "
              f"Threads: {proc_info['num_threads']} | "
              f"Status: {proc_info['status']}", end='', flush=True)
        
        time.sleep(5)  # Check every 5 seconds
    
    print(f"\n\n📊 Final Status:")
    print(f"   Episodes completed: {episode_count:,}")
    print(f"   Average speed: {episodes_per_sec:.1f} episodes/second")
    print(f"   Memory usage: {proc_info['memory_mb']:.1f} MB")
    print(f"   CPU usage: {proc_info['cpu_percent']:.1f}%")
    
    # Estimate completion time
    if episodes_per_sec > 0:
        remaining_episodes = 1000000 - episode_count
        eta_seconds = remaining_episodes / episodes_per_sec
        eta_hours = eta_seconds / 3600
        print(f"   ETA: {eta_hours:.1f} hours for 1M episodes")

def check_logs():
    """Check training logs"""
    log_file = Path('./micro_training.log')
    if log_file.exists():
        print(f"\n📝 Recent log entries:")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:  # Last 10 lines
                print(f"   {line.strip()}")

if __name__ == '__main__':
    monitor_training()
    check_logs()
