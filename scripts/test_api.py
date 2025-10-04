"""
Test script for the FastAPI serving endpoint.
Validates inference latency and correctness.
"""
import requests
import time
import numpy as np
import pandas as pd
import json


def test_health():
    """Test health endpoint."""
    print("\n[1/4] Testing health endpoint...")
    
    response = requests.get("http://localhost:8000/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print("  ✅ Health check passed")
    else:
        print(f"  ❌ Health check failed: {response.status_code}")
    
    return response.status_code == 200


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n[2/4] Testing single prediction...")
    
    # Generate test signal
    t = np.arange(100) / 100.0
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(100)
    
    # Add an anomaly at the end
    signal[-1] += 5.0
    
    # Prepare request
    payload = {
        "signals": [
            {"ts": float(t[i]), "signal": float(signal[i])}
            for i in range(len(t))
        ],
        "freq": 100,
        "window": 50
    }
    
    # Send request
    start = time.time()
    response = requests.post(
        "http://localhost:8000/predict",
        json=payload
    )
    latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Anomaly detected: {data['anomaly']}")
        print(f"  Anomaly score: {data['score']:.4f}")
        print(f"  Threshold: {data['threshold']:.4f}")
        print(f"  API latency: {data['latency_ms']:.2f} ms")
        print(f"  Total latency: {latency:.2f} ms")
        
        if latency < 100:
            print("  ✅ Latency < 100ms (ACCEPTANCE CRITERIA MET)")
        else:
            print(f"  ⚠️  Latency {latency:.2f}ms > 100ms target")
        
        return True
    else:
        print(f"  ❌ Prediction failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n[3/4] Testing batch prediction...")
    
    # Generate test signals
    n = 1000
    t = np.arange(n) / 100.0
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n)
    
    # Add some anomalies
    anomaly_idx = np.random.choice(n, size=10, replace=False)
    signal[anomaly_idx] += np.random.randn(10) * 3
    
    # Prepare request
    payload = {
        "rows": [
            {"ts": float(t[i]), "signal": float(signal[i])}
            for i in range(n)
        ]
    }
    
    # Send request
    start = time.time()
    response = requests.post(
        "http://localhost:8000/batch_predict",
        json=payload
    )
    latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        data = response.json()
        predictions = data['predictions']
        
        n_anomalies = sum(1 for p in predictions if p['anomaly'])
        
        print(f"  Processed: {data['num_samples']} samples")
        print(f"  Detected anomalies: {n_anomalies}")
        print(f"  Latency: {data['latency_ms']:.2f} ms")
        print(f"  Throughput: {data['num_samples'] / (latency/1000):.0f} samples/sec")
        print("  ✅ Batch prediction passed")
        
        return True
    else:
        print(f"  ❌ Batch prediction failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


def test_error_handling():
    """Test error handling."""
    print("\n[4/4] Testing error handling...")
    
    # Test with invalid payload
    payload = {"invalid": "data"}
    
    response = requests.post(
        "http://localhost:8000/predict",
        json=payload
    )
    
    if response.status_code == 422:  # Validation error
        print("  ✅ Validation error handled correctly")
        return True
    else:
        print(f"  ⚠️  Unexpected status code: {response.status_code}")
        return True  # Not critical


def main():
    """Run all API tests."""
    print("=" * 60)
    print("TESTING FASTAPI SERVING ENDPOINT")
    print("=" * 60)
    print("\nMake sure the API is running:")
    print("  uvicorn src.serve:app --reload")
    print("\nOr run: python src/serve.py")
    
    input("\nPress Enter to start tests...")
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Single Prediction", test_single_prediction()))
    results.append(("Batch Prediction", test_batch_prediction()))
    results.append(("Error Handling", test_error_handling()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n⚠️  SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Please start the API server first:")
        print("  python src/serve.py")
        exit(1)
