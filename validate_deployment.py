#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-deployment Validation Script
배포 전 검증 스크립트 - 3-Tier Fallback Architecture
"""

import os
import sys
import yaml
import requests
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """배포 전 검증 클래스"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> bool:
        """모든 검증 실행"""
        logger.info("Starting comprehensive deployment validation...")
        
        # Configuration validations
        self.validate_yaml_config()
        self.validate_python_config()
        self.validate_port_alignment()
        
        # Environment validations
        self.validate_api_keys()
        self.validate_file_structure()
        self.validate_requirements()
        
        # Architecture validations
        self.validate_fallback_config()
        self.validate_example_questions()
        
        return self.generate_report()
    
    def validate_yaml_config(self):
        """YAML 설정 파일 검증"""
        logger.info("Validating YAML configuration...")
        
        yaml_path = Path("vessl_configs/run_robust_fallback.yaml")
        if not yaml_path.exists():
            self.errors.append(f"YAML config file not found: {yaml_path}")
            return
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['name', 'description', 'import', 'resources', 'run', 'ports', 'env']
            for section in required_sections:
                if section not in config:
                    self.errors.append(f"Missing required section in YAML: {section}")
            
            # Validate ports configuration
            ports = config.get('ports', [])
            port_numbers = [port.get('port') for port in ports]
            
            if 7860 not in port_numbers:
                self.errors.append("Gradio port 7860 not configured in YAML")
            if 8000 not in port_numbers:
                self.errors.append("vLLM port 8000 not configured in YAML")
            
            # Validate environment variables
            env_vars = config.get('env', {})
            required_env_vars = [
                'VLLM_API_URL',
                'GRADIO_SERVER_PORT',
                'ENABLE_OPENAI_FALLBACK'
            ]
            
            for var in required_env_vars:
                if var not in env_vars:
                    self.warnings.append(f"Recommended env var not set in YAML: {var}")
            
            # Check vLLM API URL alignment
            vllm_url = env_vars.get('VLLM_API_URL', '')
            if 'localhost:8000' not in vllm_url:
                self.errors.append(f"VLLM_API_URL should contain 'localhost:8000', got: {vllm_url}")
            
            self.validation_results['yaml_config'] = 'PASS'
            logger.info("✓ YAML configuration validation passed")
            
        except Exception as e:
            self.errors.append(f"YAML validation error: {e}")
            self.validation_results['yaml_config'] = 'FAIL'
    
    def validate_python_config(self):
        """Python 설정 파일 검증"""
        logger.info("Validating Python configuration...")
        
        config_path = Path("vessl_code_/src/config.py")
        if not config_path.exists():
            self.errors.append(f"Python config file not found: {config_path}")
            return
        
        try:
            # Read config file content
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for port alignment
            if 'localhost:8088' in content:
                self.errors.append("config.py still contains port 8088 - should be 8000")
            
            if 'localhost:8000' not in content:
                self.warnings.append("config.py should contain 'localhost:8000' for vLLM")
            
            # Check for robust client import capability
            robust_client_path = Path("vessl_code_/src/robust_llm_client.py")
            if not robust_client_path.exists():
                self.warnings.append("Robust LLM client not found - fallback capabilities limited")
            
            self.validation_results['python_config'] = 'PASS'
            logger.info("✓ Python configuration validation passed")
            
        except Exception as e:
            self.errors.append(f"Python config validation error: {e}")
            self.validation_results['python_config'] = 'FAIL'
    
    def validate_port_alignment(self):
        """포트 정렬 검증"""
        logger.info("Validating port alignment...")
        
        # Check YAML ports
        yaml_path = Path("vessl_configs/run_robust_fallback.yaml")
        yaml_ports = []
        
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                yaml_ports = [port.get('port') for port in config.get('ports', [])]
            except:
                pass
        
        # Check if critical ports are aligned
        expected_ports = {8000: 'vLLM', 7860: 'Gradio'}
        for port, service in expected_ports.items():
            if port not in yaml_ports:
                self.errors.append(f"{service} port {port} not configured in YAML")
        
        self.validation_results['port_alignment'] = 'PASS' if not self.errors else 'FAIL'
        logger.info("✓ Port alignment validation completed")
    
    def validate_api_keys(self):
        """API 키 검증"""
        logger.info("Validating API key configuration...")
        
        # Check environment variable
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            self.warnings.append("OPENAI_API_KEY not set in environment - fallback mode may not work")
        
        # Check API key loader
        api_key_loader_path = Path("vessl_code_/src/api_key_loader.py")
        if not api_key_loader_path.exists():
            self.warnings.append("API key loader not found - may affect fallback functionality")
        
        self.validation_results['api_keys'] = 'PASS'
        logger.info("✓ API key validation completed")
    
    def validate_file_structure(self):
        """파일 구조 검증"""
        logger.info("Validating file structure...")
        
        required_files = [
            "vessl_code_/src/app.py",
            "vessl_code_/src/config.py",
            "vessl_code_/src/llm_client.py",
            "vessl_code_/run_app.py",
            "vessl_code_/requirements.txt",
            "vessl_configs/run_robust_fallback.yaml"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                self.errors.append(f"Required file not found: {file_path}")
        
        # Check for example questions
        example_path = Path("EXAMPLE.md")
        if not example_path.exists():
            self.warnings.append("EXAMPLE.md not found - testing may be limited")
        
        self.validation_results['file_structure'] = 'PASS' if not any('Required file' in e for e in self.errors) else 'FAIL'
        logger.info("✓ File structure validation completed")
    
    def validate_requirements(self):
        """요구사항 파일 검증"""
        logger.info("Validating requirements...")
        
        req_path = Path("vessl_code_/requirements.txt")
        if not req_path.exists():
            self.errors.append("requirements.txt not found")
            return
        
        try:
            with open(req_path, 'r') as f:
                requirements = f.read()
            
            critical_packages = [
                'transformers',
                'vllm',
                'gradio',
                'chromadb',
                'requests'
            ]
            
            for package in critical_packages:
                if package.lower() not in requirements.lower():
                    self.warnings.append(f"Critical package may be missing: {package}")
            
            self.validation_results['requirements'] = 'PASS'
            logger.info("✓ Requirements validation completed")
            
        except Exception as e:
            self.errors.append(f"Requirements validation error: {e}")
            self.validation_results['requirements'] = 'FAIL'
    
    def validate_fallback_config(self):
        """폴백 설정 검증"""
        logger.info("Validating fallback configuration...")
        
        # Check if OpenAI client exists
        openai_client_path = Path("vessl_code_/src/llm_client_openai.py")
        if not openai_client_path.exists():
            self.warnings.append("OpenAI client not found - Tier 2 fallback unavailable")
        
        # Check robust client
        robust_client_path = Path("vessl_code_/src/robust_llm_client.py")
        if not robust_client_path.exists():
            self.warnings.append("Robust client not found - enhanced fallback unavailable")
        
        self.validation_results['fallback_config'] = 'PASS'
        logger.info("✓ Fallback configuration validation completed")
    
    def validate_example_questions(self):
        """예제 질문 검증"""
        logger.info("Validating example questions...")
        
        example_path = Path("EXAMPLE.md")
        if not example_path.exists():
            self.warnings.append("EXAMPLE.md not found")
            return
        
        try:
            with open(example_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count questions
            question_count = content.count('[') - content.count('[[')  # Simple heuristic
            if question_count < 4:
                self.warnings.append(f"Found only {question_count} example questions, recommended: 5+")
            
            # Check for Korean content
            if '다산에듀' not in content:
                self.warnings.append("Example questions may not match expected content")
            
            self.validation_results['example_questions'] = 'PASS'
            logger.info("✓ Example questions validation completed")
            
        except Exception as e:
            self.warnings.append(f"Example questions validation error: {e}")
            self.validation_results['example_questions'] = 'PARTIAL'
    
    def generate_report(self) -> bool:
        """검증 결과 보고서 생성"""
        logger.info("\n" + "="*50)
        logger.info("DEPLOYMENT VALIDATION REPORT")
        logger.info("="*50)
        
        # Summary
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results.values() if result == 'PASS')
        
        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {total_checks - passed_checks}")
        logger.info("")
        
        # Detailed results
        for check, result in self.validation_results.items():
            status_icon = "✓" if result == 'PASS' else "⚠" if result == 'PARTIAL' else "✗"
            logger.info(f"{status_icon} {check.replace('_', ' ').title()}: {result}")
        
        # Errors
        if self.errors:
            logger.error("\nCRITICAL ERRORS (Must Fix):")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")
        
        # Warnings
        if self.warnings:
            logger.warning("\nWARNINGS (Recommended to Fix):")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")
        
        # Final verdict
        deployment_ready = len(self.errors) == 0
        logger.info("\n" + "="*50)
        if deployment_ready:
            logger.info("✅ DEPLOYMENT READY")
            logger.info("All critical validations passed. Deployment can proceed.")
        else:
            logger.error("❌ DEPLOYMENT NOT READY")
            logger.error("Critical errors found. Please fix before deployment.")
        
        logger.info("="*50)
        
        return deployment_ready
    
    def create_deployment_checklist(self):
        """배포 체크리스트 생성"""
        checklist_path = Path("deployment_checklist.md")
        
        checklist_content = f"""# Deployment Checklist - 3-Tier Fallback Architecture

## Pre-deployment Validation Results
Generated: {os.popen('date').read().strip()}

### Critical Items
{"✅ All critical validations passed" if len(self.errors) == 0 else "❌ Critical errors found"}

### Validation Summary
"""
        for check, result in self.validation_results.items():
            status = "✅" if result == 'PASS' else "⚠️" if result == 'PARTIAL' else "❌"
            checklist_content += f"- {status} {check.replace('_', ' ').title()}\n"
        
        if self.errors:
            checklist_content += "\n### Critical Errors to Fix\n"
            for error in self.errors:
                checklist_content += f"- ❌ {error}\n"
        
        if self.warnings:
            checklist_content += "\n### Warnings (Recommended)\n"
            for warning in self.warnings:
                checklist_content += f"- ⚠️ {warning}\n"
        
        checklist_content += """
## Deployment Steps
1. ✅ Run pre-deployment validation
2. 🔄 Execute VESSL deployment: `vessl run create -f vessl_configs/run_robust_fallback.yaml`
3. 👀 Monitor logs: `vessl run logs <run-id> -f`
4. 🧪 Test with example questions from EXAMPLE.md
5. 🎯 Verify all 3 tiers are functional
6. 🛑 Terminate deployment after testing: `vessl run terminate <run-id>`

## Architecture Overview
- **Tier 1 (Primary)**: vLLM server on port 8000
- **Tier 2 (Fallback)**: OpenAI API with stored key
- **Tier 3 (Hybrid)**: Minimal functionality mode

## Monitoring Points
- vLLM server health: http://localhost:8000/health
- Gradio interface: http://localhost:7860
- Model endpoint: http://localhost:8000/v1/models
- Test completion: API functional test
"""
        
        with open(checklist_path, 'w', encoding='utf-8') as f:
            f.write(checklist_content)
        
        logger.info(f"Deployment checklist created: {checklist_path}")


def main():
    """메인 실행 함수"""
    validator = DeploymentValidator()
    
    # Run all validations
    is_ready = validator.validate_all()
    
    # Create deployment checklist
    validator.create_deployment_checklist()
    
    # Exit with appropriate code
    sys.exit(0 if is_ready else 1)


if __name__ == "__main__":
    main()