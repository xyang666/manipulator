.PHONY: all clean view

all: main.pdf

main.pdf: main.tex references.bib iclr2026_conference.sty
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf

view: main.pdf
	xdg-open main.pdf 2>/dev/null || open main.pdf 2>/dev/null || echo "Please open main.pdf manually"
