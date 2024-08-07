import pdfplumber
import os

# 指定要处理的 PDF 文件列表
pdf_files = [
    "/home/jisheng/Desktop/CS-Professor/res/Abraham_Silberschatz_Greg_Gagne_Peter_B._Galvin_-_Operating_System_Concepts-Wiley_2018.pdf",
    "/home/jisheng/Desktop/CS-Professor/res/ComputerOrganizationAndDesign5thEdition2014.pdf",
    "/home/jisheng/Desktop/CS-Professor/res/M. Morris Mano, Charles Kime - Logic and Computer Design Fundamentals (4th Edition) Solutions textbook.-Prentice Hall (2007).pdf",
    "/home/jisheng/Desktop/CS-Professor/res/Ramez_Elmasri_Shamkant_B._Navathe_-_Fundamentals_of_Database_Systems_6th_Edition_-Addison-Wesley_2010.pdf",
    "/home/jisheng/Desktop/CS-Professor/res/Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein Introduction to Algorithms, Third Edition  2009.pdf"
]

# 设置输出文件夹
output_folder = "/home/jisheng/Desktop/CS-Professor/res/extracted_texts/"
os.makedirs(output_folder, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """从 PDF 文件中提取文本"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def process_pdf(pdf_path, output_folder):
    """处理单个 PDF 文件并保存提取的文本"""
    file_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
    print(f"正在处理: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, "w") as text_file:
        text_file.write(text)
    print(f"完成: {pdf_path}")

def process_pdfs(pdf_files, output_folder):
    """处理指定的所有 PDF 文件"""
    total_files = len(pdf_files)
    
    if total_files == 0:
        print("没有找到 PDF 文件。")
        return

    print(f"找到 {total_files} 个 PDF 文件。")
    for i, pdf_file in enumerate(pdf_files):
        print(f"处理文件 {i + 1} / {total_files}: {pdf_file}")
        process_pdf(pdf_file, output_folder)
    print("所有文件处理完成。")

# 执行处理
process_pdfs(pdf_files, output_folder)
