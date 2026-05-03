# Build the paper. Run `make` from the paper/ directory.
# Requires latexmk + pdflatex + bibtex (MiKTeX or TeX Live both work).

LATEXMK ?= latexmk
JOBNAME  := main

.PHONY: all clean cleanall watch

all: $(JOBNAME).pdf

$(JOBNAME).pdf: $(JOBNAME).tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*.pdf)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(JOBNAME).tex

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(JOBNAME).tex

clean:
	$(LATEXMK) -c

cleanall:
	$(LATEXMK) -C
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz
