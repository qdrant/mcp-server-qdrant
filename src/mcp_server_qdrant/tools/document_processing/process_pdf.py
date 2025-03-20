"""
PDF Processing tool for Qdrant MCP server.
"""
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from mcp.server.fastmcp import Context

from mcp_server_qdrant.tools.data_processing.chunk_and_process import chunk_and_process


async def process_pdf(
    ctx: Context,
    pdf_path: str,
    collection: Optional[str] = None,
    extract_tables: bool = False,
    extract_images: bool = False,
) -> Dict[str, Any]:
    """
    Process a PDF file, extract content, and optionally store in Qdrant.
    
    Note: This implementation requires additional libraries for PDF processing:
    - PyPDF2 or pdfplumber for text extraction
    - pdf2image for image extraction
    - camelot-py or tabula-py for table extraction
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    pdf_path : str
        Path to the PDF file
    collection : str, optional
        Collection to store the document chunks in
    extract_tables : bool, default=False
        Whether to extract tables from the PDF
    extract_images : bool, default=False
        Whether to extract images from the PDF
        
    Returns:
    --------
    Dict[str, Any]
        Processing results
    """
    await ctx.debug(f"Processing PDF: {pdf_path}")
    
    try:
        # Verify the file exists
        if not os.path.exists(pdf_path):
            return {
                "status": "error",
                "message": f"PDF file not found: {pdf_path}"
            }
        
        # Import PDF processing libraries
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_path)
        except ImportError:
            await ctx.debug("PyPDF2 not available, trying pdfplumber")
            try:
                import pdfplumber
                pdf_reader = pdfplumber.open(pdf_path)
            except ImportError:
                return {
                    "status": "error",
                    "message": "No PDF processing library available. Please install PyPDF2 or pdfplumber"
                }
        
        # Extract metadata
        try:
            if isinstance(pdf_reader, PyPDF2.PdfReader):
                metadata = pdf_reader.metadata
                if metadata:
                    metadata = {k.lower().replace('/', '_'): v for k, v in metadata.items()}
                else:
                    metadata = {}
                num_pages = len(pdf_reader.pages)
            else:  # pdfplumber
                metadata = pdf_reader.metadata
                if metadata:
                    metadata = {k.lower().replace('/', '_'): v for k, v in metadata.items()}
                else:
                    metadata = {}
                num_pages = len(pdf_reader.pages)
        except Exception as e:
            await ctx.debug(f"Error extracting metadata: {str(e)}")
            metadata = {}
            num_pages = 0
        
        # Extract text
        try:
            if isinstance(pdf_reader, PyPDF2.PdfReader):
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n\n"
            else:  # pdfplumber
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n\n"
        except Exception as e:
            await ctx.debug(f"Error extracting text: {str(e)}")
            text_content = ""
        
        # Process tables if requested
        tables = []
        if extract_tables:
            try:
                # Try camelot first
                try:
                    import camelot
                    tables_df = camelot.read_pdf(pdf_path)
                    for i, table in enumerate(tables_df):
                        tables.append({
                            "table_id": i,
                            "content": table.df.to_dict(),
                            "page": table.page
                        })
                except ImportError:
                    # Try tabula
                    try:
                        import tabula
                        tables_df = tabula.read_pdf(pdf_path, pages='all')
                        for i, table in enumerate(tables_df):
                            tables.append({
                                "table_id": i,
                                "content": table.to_dict(),
                                "page": "unknown"  # tabula doesn't provide page info easily
                            })
                    except ImportError:
                        await ctx.debug("No table extraction library available. Please install camelot-py or tabula-py")
            except Exception as e:
                await ctx.debug(f"Error extracting tables: {str(e)}")
        
        # Process images if requested
        images = []
        if extract_images:
            try:
                import pdf2image
                import tempfile
                
                # Create temp directory for images
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_images = pdf2image.convert_from_path(pdf_path)
                    for i, image in enumerate(pdf_images):
                        image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                        image.save(image_path, "PNG")
                        images.append({
                            "image_id": i,
                            "page": i+1,
                            "path": image_path
                        })
            except ImportError:
                await ctx.debug("pdf2image not available. Please install pdf2image for image extraction")
            except Exception as e:
                await ctx.debug(f"Error extracting images: {str(e)}")
        
        # Create document metadata
        doc_metadata = {
            "source_path": pdf_path,
            "source_type": "pdf",
            "processed_at": datetime.now().isoformat(),
            "num_pages": num_pages,
            "has_tables": len(tables) > 0,
            "has_images": len(images) > 0,
            **metadata
        }
        
        # Process for vector search if collection specified
        chunks_result = None
        if collection and text_content:
            chunks_result = await chunk_and_process(
                ctx=ctx,
                text=text_content,
                collection=collection,
                metadata=doc_metadata
            )
        
        return {
            "status": "success",
            "metadata": doc_metadata,
            "text_length": len(text_content),
            "tables_count": len(tables),
            "images_count": len(images),
            "chunks_result": chunks_result
        }
    except Exception as e:
        await ctx.debug(f"Error processing PDF: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to process PDF: {str(e)}"
        }
