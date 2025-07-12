# KW Environment Refactoring

## Overview

The KW environments have been refactored from a single large file (`kw_env.py`) into separate, focused files for better maintainability and organization.

## File Structure

### Before Refactoring
```
kw_env.py (777 lines) - Contains all 4 environment classes
```

### After Refactoring
```
kw_and_env.py     - KWAndEnv class (AND function)
kw_bip_env.py     - KWBIPEnv class (Boolean Inner Product)
kw_ip_env.py      - KWIPEnv class (Inner Product modulo 2)
kwvp_env.py       - KWVPEnv class (Vector Product)
kw_env.py         - Main import file (backward compatibility)
test_environments.py - Test script to verify functionality
```

## Environment Classes

### 1. KWAndEnv (`kw_and_env.py`)
- **Function**: AND(x) = x₁ ∧ x₂ ∧ ... ∧ xₙ
- **Player-0**: Holds x s.t. AND(x) = 1 (all bits are 1)
- **Player-1**: Holds y s.t. AND(y) = 0 (at least one bit is 0)
- **Goal**: Find index i where x_i = 1 and y_i = 0

### 2. KWBIPEnv (`kw_bip_env.py`)
- **Function**: Boolean Inner Product BIP(x₁, x₂) = OR of element-wise AND
- **Player-0**: Holds x s.t. BIP(x₁, x₂) = 1 (minterm)
- **Player-1**: Holds y s.t. BIP(y₁, y₂) = 0 (maxterm)
- **Goal**: Find index i where x_i = 1 and y_i = 0

### 3. KWIPEnv (`kw_ip_env.py`)
- **Function**: Inner Product modulo 2 IP(x₁, x₂) = (x₁ · x₂) mod 2
- **Player-0**: Holds x s.t. IP(x₁, x₂) = 1 (odd inner product)
- **Player-1**: Holds y s.t. IP(y₁, y₂) = 0 (even inner product)
- **Goal**: Find index i where x_i ≠ y_i

### 4. KWVPEnv (`kwvp_env.py`)
- **Function**: Vector Product VP(x₁, x₂) = OR of element-wise AND
- **Player-0**: Holds x s.t. VP(x₁, x₂) = 1 (minterm)
- **Player-1**: Holds y s.t. VP(y₁, y₂) = 0 (maxterm)
- **Goal**: Find index i where x_i = 1 and y_i = 0

## Backward Compatibility

The original `kw_env.py` file has been replaced with a simple import file that maintains backward compatibility:

```python
# Import all KW environment classes from their individual files
from kw_and_env import KWAndEnv
from kw_bip_env import KWBIPEnv
from kw_ip_env import KWIPEnv
from kwvp_env import KWVPEnv

# Re-export all classes for backward compatibility
__all__ = ['KWAndEnv', 'KWBIPEnv', 'KWIPEnv', 'KWVPEnv']
```

This means existing code that imports from `kw_env` will continue to work without modification.

## Usage

### Import from main file (recommended for existing code)
```python
from kw_env import KWAndEnv, KWBIPEnv, KWIPEnv, KWVPEnv
```

### Import individual files (recommended for new code)
```python
from kw_and_env import KWAndEnv
from kw_bip_env import KWBIPEnv
from kw_ip_env import KWIPEnv
from kwvp_env import KWVPEnv
```

## Testing

Run the test script to verify all environments work correctly:

```bash
source kwrl_env/bin/activate
python test_environments.py
```

## Benefits of Refactoring

1. **Better Organization**: Each environment is in its own focused file
2. **Easier Maintenance**: Changes to one environment don't affect others
3. **Improved Readability**: Smaller files are easier to understand
4. **Better Version Control**: Changes to different environments can be tracked separately
5. **Modular Development**: New environments can be added without cluttering existing code
6. **Backward Compatibility**: Existing code continues to work without modification

## Technical Details

- All files include proper type hints with `Optional[int]` for the `last_action` dictionary
- Each environment maintains the same interface and behavior as before
- The refactoring fixed linter errors related to type annotations
- All environments pass the test suite 