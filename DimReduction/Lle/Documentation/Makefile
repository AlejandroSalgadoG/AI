Article_lle.pdf: Article_lle.tex Article_lle.bib
	pdflatex Article_lle.tex
	bibtex Article_lle
	pdflatex Article_lle.tex
	pdflatex Article_lle.tex
	pdflatex Article_lle.tex

clean:
	rm -f Article_lle.pdf
	rm -f *.nav *.snm *.toc *.out *.dvi *.blg *.bbl *.aux *.log

exe:
	evince Article_lle.pdf &
