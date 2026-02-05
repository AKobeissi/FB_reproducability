"""
Test script to verify page indexing consistency between:
1. PyMuPDFLoader (langchain) - 0-indexed
2. extract_pages_from_pdf (custom) - 0-indexed
3. FinanceBench ground truth - 0-indexed
"""
import logging
from pathlib import Path
from src.ingestion.page_processor import extract_pages_from_pdf
from src.ingestion.data_loader import FinanceBenchLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("PAGE INDEXING VERIFICATION (0-indexed)")
    logger.info("="*80)
    
    # Load dataset
    loader = FinanceBenchLoader()
    df = loader.load_data()
    
    # Test with first few samples
    for sample_idx in range(min(3, len(df))):
        sample = df.iloc[sample_idx]
        doc_name = sample['doc_name']
        evidence = sample['evidence']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {sample_idx}:")
        logger.info(f"  Doc: {doc_name}")
        logger.info(f"  Question: {sample['question'][:100]}...")
        
        # Debug: Show evidence structure
        logger.info(f"\n  Evidence field:")
        logger.info(f"    Type: {type(evidence)}")
        logger.info(f"    Value: {evidence}")
        
        # Convert to list if it's array-like (pandas Series, numpy array, etc.)
        if not isinstance(evidence, list):
            try:
                evidence = list(evidence)
                logger.info(f"    Converted to list, length: {len(evidence)}")
            except:
                logger.warning(f"  Cannot convert evidence to list!")
                logger.info(f"  All sample keys: {list(sample.keys())}")
                continue
        
        if len(evidence) == 0:
            logger.warning(f"  Evidence list is empty!")
            continue
        
        ev = evidence[0]
        gold_page = ev.get('evidence_page_num') or ev.get('page')
        logger.info(f"  Gold evidence page: {gold_page} (0-indexed)")
        logger.info(f"  PDF viewer would show: page {int(gold_page) + 1}")
        logger.info(f"  Evidence text: {ev.get('evidence_text', '')[:150]}...")
        
        # Extract pages using custom function (now uses PyMuPDFLoader)
        pdf_path = Path("pdfs") / f"{doc_name}.pdf"
        if not pdf_path.exists():
            # Try case-insensitive
            pdf_dir = Path("pdfs")
            for p in pdf_dir.iterdir():
                if p.stem.lower() == doc_name.lower():
                    pdf_path = p
                    break
        
        if pdf_path.exists():
            logger.info(f"\n  Extracting from: {pdf_path.name}")
            pages = extract_pages_from_pdf(pdf_path, doc_name)
            logger.info(f"  Total pages extracted: {len(pages)}")
            
            if pages:
                logger.info(f"\n  First page (0):")
                logger.info(f"    page field: {pages[0]['page']}")
                logger.info(f"    text length: {len(pages[0]['text'])}")
                
                # Check if gold evidence page exists
                if gold_page is not None:
                    gold_page_int = int(gold_page)
                    if gold_page_int < len(pages):
                        gold_page_text = pages[gold_page_int]['text']
                        logger.info(f"\n  ✓ Gold evidence page {gold_page_int} (PDF viewer: {gold_page_int + 1}):")
                        logger.info(f"    Text length: {len(gold_page_text)}")
                        logger.info(f"    Text preview: {gold_page_text[:200]}...")
                        
                        # Check if evidence text is in the page
                        evidence_text = ev.get('evidence_text', '')
                        if evidence_text:
                            # Normalize for comparison
                            norm_page = ' '.join(gold_page_text.lower().split())
                            norm_evidence = ' '.join(evidence_text.lower().split())
                            if norm_evidence in norm_page:
                                logger.info(f"    ✓✓ Evidence text FOUND in extracted page!")
                            else:
                                logger.warning(f"    ✗ Evidence text NOT FOUND in extracted page")
                                logger.info(f"    Evidence to find: {evidence_text[:100]}...")
                    else:
                        logger.error(f"\n  ✗ Gold page {gold_page_int} OUT OF RANGE (max={len(pages)-1})")
        else:
            logger.error(f"  PDF not found: {pdf_path}")
    
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION COMPLETE")
    logger.info("Expected: Gold pages are 0-indexed (page 59 in data = page 60 in PDF viewer)")
    logger.info("="*80)

if __name__ == "__main__":
    main()
