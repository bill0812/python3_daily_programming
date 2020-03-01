from fpdf import FPDF 
   
# save FPDF() class into  
# a variable pdf 
pdf = FPDF()    
   
# Add a page 
pdf.add_page() 
   
# set style and size of font  
# that you want in the pdf 
pdf.set_font("Arial", size = 15) 
  
# open the text file in read mode 
f = open("flops_param.txt", "r") 
  
# insert the texts in pdf 
for x in f: 
    pdf.cell(w=0, h=40, txt = x, ln = 1, border=1) 
   
# save the pdf with name .pdf 
pdf.output("flops_param.pdf")    