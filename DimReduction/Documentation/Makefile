poster.pdf: poster.tex poster.bib
	pdflatex poster.tex
	bibtex poster
	pdflatex poster.tex
	pdflatex poster.tex
	pdflatex poster.tex

clean:
	rm -f poster.pdf
	rm -f *.nav *.snm *.toc *.out *.dvi *.blg *.bbl *.aux *.log *.ps

exe:
	evince poster.pdf &
