import time
import psutil
import GPUtil
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class PerformanceBenchmark:
    def __init__(self):
        self.models = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt',
            'medium': 'yolov8m.pt'
        }
        
    def benchmark_models(self):
        """Benchmark different YOLO model sizes"""
        results = {}
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        for model_name, model_path in self.models.items():
            print(f"\nBenchmarking {model_name} model...")
            
            # Load model
            model = YOLO(model_path)
            
            # Warmup
            for _ in range(5):
                model(test_image, verbose=False)
            
            # Benchmark
            times = []
            cpu_usage = []
            memory_usage = []
            
            for i in range(50):
                # Monitor system resources
                cpu_before = psutil.cpu_percent()
                memory_before = psutil.virtual_memory().percent
                
                start_time = time.time()
                results_model = model(test_image, verbose=False)
                end_time = time.time()
                
                cpu_after = psutil.cpu_percent()
                memory_after = psutil.virtual_memory().percent
                
                times.append(end_time - start_time)
                cpu_usage.append(max(cpu_after, cpu_before))
                memory_usage.append(max(memory_after, memory_before))
            
            # Calculate statistics
            results[model_name] = {
                'avg_inference_time': np.mean(times),
                'min_inference_time': np.min(times),
                'max_inference_time': np.max(times),
                'fps': 1.0 / np.mean(times),
                'avg_cpu_usage': np.mean(cpu_usage),
                'avg_memory_usage': np.mean(memory_usage),
            }
        
        # Print results
        self.print_benchmark_results(results)
        return results
    
    def print_benchmark_results(self, results):
        """Print benchmark results in a formatted table"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        print(f"{'Model':<10} {'FPS':<8} {'Inf.Time':<12} {'CPU%':<8} {'Memory%':<10}")
        print("-"*60)
        
        for model_name, stats in results.items():
            print(f"{model_name:<10} {stats['fps']:<8.1f} {stats['avg_inference_time']*1000:<12.1f}ms {stats['avg_cpu_usage']:<8.1f} {stats['avg_memory_usage']:<10.1f}")
        
        print("-"*60)
        print("Recommendation for drone deployment:")
        
        # Find best model for drone use
        best_fps = max(results.values(), key=lambda x: x['fps'])
        best_model = [name for name, stats in results.items() if stats['fps'] == best_fps['fps']][0]
        
        print(f"Best performance: {best_model} model")
        print(f"Expected drone FPS: {best_fps['fps']:.1f}")

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.benchmark_models()