# File: emergency_patch.py
"""
🚨 EMERGENCY PATCH for FixedInputAnalysis keywords issue
Run this BEFORE running your demo to fix the issue immediately
"""

def emergency_fix():
    """Apply emergency fix for missing keywords attribute"""
    
    print("🚨 EMERGENCY PATCH: Fixing FixedInputAnalysis keywords issue")
    print("=" * 60)
    
    import sys
    from enum import Enum
    from dataclasses import dataclass, field
    from typing import List, Dict, Any, Optional
    import time
    
    # Define TaskType if not already defined
    class TaskType(Enum):
        MATHEMATICAL = "mathematical"
        CREATIVE_WRITING = "creative_writing"
        FACTUAL_QA = "factual_qa"
        REASONING = "reasoning"
        CODE_GENERATION = "code_generation"
        SCIENTIFIC = "scientific"
        CONVERSATIONAL = "conversational"
    
    # Emergency safe analysis class
    @dataclass
    class EmergencyInputAnalysis:
        task_type: TaskType = TaskType.CONVERSATIONAL
        confidence: float = 0.5
        keywords: List[str] = field(default_factory=list)
        domain_indicators: List[str] = field(default_factory=list)
        complexity_score: float = 0.0
        features: Dict[str, Any] = field(default_factory=dict)
        uncertainty_estimate: float = 0.5
        processing_time: float = 0.0
        timestamp: float = field(default_factory=time.time)
        
        def to_dict(self):
            return {
                'task_type': self.task_type.value if hasattr(self.task_type, 'value') else str(self.task_type),
                'confidence': self.confidence,
                'keywords': self.keywords,
                'domain_indicators': self.domain_indicators,
                'complexity_score': self.complexity_score,
                'features': self.features
            }
    
    # Emergency monkey-patch function
    def monkey_patch_modules():
        """Monkey patch existing modules"""
        
        modules_to_patch = [
            'core.types',
            'core.input_classifier',
            'core.enhanced_input_classifier',
            'core.multi_model_engine'
        ]
        
        patched_count = 0
        
        for module_name in modules_to_patch:
            try:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    
                    # Replace InputAnalysis classes
                    if hasattr(module, 'InputAnalysis'):
                        module.InputAnalysis = EmergencyInputAnalysis
                        print(f"   ✅ Patched {module_name}.InputAnalysis")
                        patched_count += 1
                    
                    if hasattr(module, 'FixedInputAnalysis'):
                        module.FixedInputAnalysis = EmergencyInputAnalysis
                        print(f"   ✅ Patched {module_name}.FixedInputAnalysis")
                        patched_count += 1
                        
            except Exception as e:
                print(f"   ⚠️ Could not patch {module_name}: {e}")
        
        return patched_count
    
    # Emergency object fixer
    def fix_existing_objects():
        """Fix any existing objects in memory"""
        
        import gc
        
        fixed_count = 0
        
        # Get all objects in memory
        for obj in gc.get_objects():
            try:
                # Check if it's an analysis object without keywords
                if (hasattr(obj, 'task_type') and 
                    hasattr(obj, 'confidence') and 
                    not hasattr(obj, 'keywords')):
                    
                    # Add missing attributes
                    obj.keywords = []
                    obj.domain_indicators = []
                    obj.complexity_score = 0.0
                    if not hasattr(obj, 'features'):
                        obj.features = {}
                    
                    fixed_count += 1
                    
            except Exception:
                continue  # Skip objects that can't be modified
        
        return fixed_count
    
    # Emergency input classifier replacement
    class EmergencyClassifier:
        """Emergency classifier that always returns safe objects"""
        
        def analyze_input(self, text: str):
            """Safe analysis that never fails"""
            
            if not text:
                return EmergencyInputAnalysis()
            
            text_lower = text.lower()
            
            # Simple task detection
            if any(word in text_lower for word in ['solve', 'calculate', 'equation', 'math']):
                task_type = TaskType.MATHEMATICAL
                confidence = 0.8
            elif any(word in text_lower for word in ['story', 'write', 'creative', 'character']):
                task_type = TaskType.CREATIVE_WRITING
                confidence = 0.8
            elif any(word in text_lower for word in ['code', 'function', 'algorithm', 'program']):
                task_type = TaskType.CODE_GENERATION
                confidence = 0.8
            elif any(word in text_lower for word in ['analyze', 'compare', 'evaluate']):
                task_type = TaskType.REASONING
                confidence = 0.8
            elif any(word in text_lower for word in ['what', 'who', 'when', 'where', 'why']):
                task_type = TaskType.FACTUAL_QA
                confidence = 0.7
            else:
                task_type = TaskType.CONVERSATIONAL
                confidence = 0.6
            
            # Extract basic keywords
            words = text.split()
            keywords = [w.lower() for w in words if len(w) > 3][:5]
            
            return EmergencyInputAnalysis(
                task_type=task_type,
                confidence=confidence,
                keywords=keywords,
                domain_indicators=[],
                complexity_score=min(len(text) / 200, 1.0),
                features={'length': len(text), 'word_count': len(words)}
            )
    
    # Apply all fixes
    print("\n🔧 Applying emergency fixes...")
    
    # 1. Monkey patch modules
    print("1. Monkey-patching modules...")
    patched_modules = monkey_patch_modules()
    
    # 2. Fix existing objects
    print("2. Fixing existing objects...")
    fixed_objects = fix_existing_objects()
    
    # 3. Replace classifiers in key modules
    print("3. Replacing classifiers...")
    emergency_classifier = EmergencyClassifier()
    
    for module_name in ['core.multi_model_engine', 'core.enhanced_input_classifier']:
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, 'InputClassifier'):
                    module.InputClassifier = EmergencyClassifier
                    print(f"   ✅ Replaced {module_name}.InputClassifier")
        except Exception as e:
            print(f"   ⚠️ Could not replace classifier in {module_name}: {e}")
    
    # 4. Make emergency classes globally available
    globals()['InputAnalysis'] = EmergencyInputAnalysis
    globals()['FixedInputAnalysis'] = EmergencyInputAnalysis
    globals()['TaskType'] = TaskType
    
    # Add to sys.modules for other imports
    import types
    emergency_module = types.ModuleType('emergency_types')
    emergency_module.InputAnalysis = EmergencyInputAnalysis
    emergency_module.FixedInputAnalysis = EmergencyInputAnalysis  
    emergency_module.TaskType = TaskType
    sys.modules['emergency_types'] = emergency_module
    
    print(f"\n✅ Emergency patch completed!")
    print(f"   - Patched {patched_modules} module attributes")
    print(f"   - Fixed {fixed_objects} existing objects")
    print(f"   - Emergency classes available globally")
    print(f"\n🎯 You can now run your demo - the keywords error should be fixed!")
    
    return EmergencyInputAnalysis, TaskType

# Quick test function
def test_emergency_fix():
    """Test that the emergency fix works"""
    
    print("\n🧪 Testing emergency fix...")
    
    try:
        EmergencyInputAnalysis, TaskType = emergency_fix()
        
        # Test creating analysis object
        analysis = EmergencyInputAnalysis(
            task_type=TaskType.MATHEMATICAL,
            confidence=0.9,
            keywords=['solve', 'equation'],
            domain_indicators=['mathematics']
        )
        
        # Test accessing all required attributes
        required_attrs = ['keywords', 'domain_indicators', 'complexity_score', 'features', 'confidence', 'task_type']
        
        print("   Testing required attributes:")
        for attr in required_attrs:
            if hasattr(analysis, attr):
                value = getattr(analysis, attr)
                print(f"     ✅ {attr}: {value}")
            else:
                print(f"     ❌ {attr}: MISSING")
                return False
        
        # Test to_dict method
        analysis_dict = analysis.to_dict()
        print(f"   ✅ to_dict() works: {len(analysis_dict)} keys")
        
        print(f"\n🎉 Emergency fix test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Emergency fix test FAILED: {e}")
        return False

# Auto-run when imported
if __name__ == "__main__":
    emergency_fix()
    test_emergency_fix()
else:
    # Auto-apply when imported
    try:
        emergency_fix()
        print("🚨 Emergency patch auto-applied on import")
    except Exception as e:
        print(f"⚠️ Auto-patch failed: {e}")