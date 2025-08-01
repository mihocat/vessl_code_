#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize Enhanced Vector Database with Multimodal Documents
멀티모달 문서로 향상된 벡터 데이터베이스 초기화
"""

import sys
import os
import json
import logging
from typing import List, Dict, Any

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_rag_system import EnhancedVectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_existing_data(json_path: str) -> List[Dict[str, Any]]:
    """기존 JSON 데이터 로드"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        logger.error(f"Failed to load data from {json_path}: {e}")
        return []


def convert_to_multimodal_format(qa_item: Dict[str, Any]) -> Dict[str, Any]:
    """기존 Q&A를 멀티모달 포맷으로 변환"""
    content = {
        'text': f"질문: {qa_item['question']}\n답변: {qa_item['answer']}",
        'formulas': [],
        'diagrams': []
    }
    
    # 수식 추출 (간단한 패턴 매칭)
    import re
    
    # LaTeX 수식 패턴
    latex_pattern = r'\$([^\$]+)\$|\\\[(.*?)\\\]'
    formulas = re.findall(latex_pattern, qa_item.get('answer', ''))
    
    for formula in formulas:
        # 튜플에서 비어있지 않은 값 선택
        formula_text = formula[0] if formula[0] else formula[1]
        if formula_text:
            content['formulas'].append(formula_text)
    
    # 메타데이터 구성
    metadata = {
        'source': qa_item.get('source', 'legacy'),
        'category': qa_item.get('category', 'general'),
        'concepts': []
    }
    
    # 개념 추출 (전기공학 키워드)
    keywords = ['전압', '전류', '저항', '전력', '회로', '변압기', '모터', '임피던스', '역률']
    text_lower = content['text'].lower()
    
    for keyword in keywords:
        if keyword in text_lower:
            metadata['concepts'].append(keyword)
    
    return content, metadata


def initialize_enhanced_db(
    json_path: str = "data/electrical_engineering_qa.json",
    persist_dir: str = "./chroma_db_enhanced"
):
    """향상된 데이터베이스 초기화"""
    logger.info("Initializing enhanced vector database...")
    
    # 1. 향상된 벡터 DB 생성
    vector_db = EnhancedVectorDatabase(persist_directory=persist_dir)
    
    # 2. 기존 데이터 로드
    qa_data = load_existing_data(json_path)
    logger.info(f"Loaded {len(qa_data)} Q&A pairs")
    
    # 3. 멀티모달 포맷으로 변환 및 추가
    success_count = 0
    for i, qa_item in enumerate(qa_data):
        try:
            content, metadata = convert_to_multimodal_format(qa_item)
            
            # 문서 추가
            doc_id = vector_db.add_multimodal_document(
                content=content,
                metadata=metadata
            )
            
            success_count += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(qa_data)} documents")
                
        except Exception as e:
            logger.error(f"Failed to add document {i}: {e}")
    
    logger.info(f"Successfully added {success_count}/{len(qa_data)} documents")
    
    # 4. 샘플 멀티모달 문서 추가
    sample_multimodal_docs = [
        {
            'content': {
                'text': '3상 전력 시스템에서 Y결선과 Δ결선의 차이점을 설명합니다.',
                'formulas': [
                    'V_{line} = \\sqrt{3} \\times V_{phase}',
                    'I_{line} = I_{phase}'
                ],
                'diagrams': [{
                    'type': 'circuit',
                    'components': [
                        {'type': 'voltage', 'label': 'Va'},
                        {'type': 'voltage', 'label': 'Vb'},
                        {'type': 'voltage', 'label': 'Vc'}
                    ]
                }]
            },
            'metadata': {
                'source': 'sample',
                'category': 'power_systems',
                'concepts': ['3상', 'Y결선', 'Δ결선', '전압', '전류']
            }
        },
        {
            'content': {
                'text': 'RLC 회로의 공진 조건과 공진 주파수 계산 방법입니다.',
                'formulas': [
                    'f_0 = \\frac{1}{2\\pi\\sqrt{LC}}',
                    'Q = \\frac{1}{R}\\sqrt{\\frac{L}{C}}'
                ],
                'diagrams': [{
                    'type': 'circuit',
                    'components': [
                        {'type': 'resistor', 'label': 'R'},
                        {'type': 'inductor', 'label': 'L'},
                        {'type': 'capacitor', 'label': 'C'}
                    ]
                }]
            },
            'metadata': {
                'source': 'sample',
                'category': 'circuit_analysis',
                'concepts': ['RLC', '공진', '주파수', 'Q팩터']
            }
        }
    ]
    
    # 샘플 문서 추가
    for doc in sample_multimodal_docs:
        try:
            doc_id = vector_db.add_multimodal_document(
                content=doc['content'],
                metadata=doc['metadata']
            )
            logger.info(f"Added sample document: {doc['metadata']['concepts']}")
        except Exception as e:
            logger.error(f"Failed to add sample document: {e}")
    
    logger.info("Enhanced database initialization complete!")
    
    # 5. 검증
    logger.info("\nVerifying database...")
    test_queries = [
        "3상 전력 계산",
        "RLC 공진 주파수",
        "변압기 원리"
    ]
    
    for query in test_queries:
        results = vector_db.search_multimodal(query, k=3)
        logger.info(f"\nQuery: '{query}' - Found {len(results)} results")
        if results:
            logger.info(f"  Top result score: {results[0].get('hybrid_score', 0):.3f}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Enhanced Vector Database")
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/electrical_engineering_qa.json",
        help="Path to existing Q&A JSON file"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db_enhanced",
        help="Directory for enhanced vector database"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing database before initialization"
    )
    
    args = parser.parse_args()
    
    # 기존 DB 삭제 (요청된 경우)
    if args.clear and os.path.exists(args.persist_dir):
        import shutil
        logger.warning(f"Clearing existing database at {args.persist_dir}")
        shutil.rmtree(args.persist_dir)
    
    # DB 초기화
    initialize_enhanced_db(
        json_path=args.json_path,
        persist_dir=args.persist_dir
    )


if __name__ == "__main__":
    main()