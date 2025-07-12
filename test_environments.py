#!/usr/bin/env python3
"""
Test script to verify that all KW environments work correctly after refactoring.
"""

import numpy as np
from kw_env import KWAndEnv, KWBIPEnv, KWIPEnv, KWVPEnv

def test_environment(env_class, name, n=4):
    """Test a single environment class."""
    print(f"\nTesting {name}...")
    
    try:
        # Create environment
        env = env_class(n=n)
        print(f"✓ Created {name} with n={n}")
        
        # Test reset
        obs0, obs1 = env.reset()
        print(f"✓ Reset successful - obs shapes: {obs0.shape}, {obs1.shape}")
        
        # Test a few steps
        for step in range(5):
            # Random actions
            action0 = np.random.randint(env.action_space_p0.n)
            action1 = np.random.randint(env.action_space_p1.n)
            
            obs, rewards, done, truncated, info = env.step((action0, action1))
            
            if done:
                print(f"✓ Episode ended after {step+1} steps")
                break
                
        print(f"✓ {name} test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error testing {name}: {e}")
        return False

def main():
    """Test all environments."""
    print("Testing KW Environments after refactoring...")
    
    # Test all environments
    environments = [
        (KWAndEnv, "KWAndEnv"),
        (KWBIPEnv, "KWBIPEnv"), 
        (KWIPEnv, "KWIPEnv"),
        (KWVPEnv, "KWVPEnv")
    ]
    
    results = []
    for env_class, name in environments:
        success = test_environment(env_class, name)
        results.append((name, success))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print("="*50)
    
    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Refactoring successful.")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
    print("="*50)

if __name__ == "__main__":
    main() 