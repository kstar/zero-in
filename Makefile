# From StackOverflow: https://stackoverflow.com/questions/23139933/how-to-write-a-makefile-to-run-pylint
PYLINT = pylint
PYLINTFLAGS = -rn --errors-only --disable=C --ignore=ui_eyepieceview.py --rcfile=pylintrc

PYTHONFILES := $(wildcard *.py)
PYTHONFILES := $(filter-out ui.py, $(PYTHONFILES))
PYTHONFILES := $(filter-out ui_eyepieceview.py, $(PYTHONFILES))

# The following are filtered out due to what seems like a bug in pylint :-(
PYTHONFILES := $(filter-out error_handler.py, $(PYTHONFILES))
PYTHONFILES := $(filter-out imageviewer.py, $(PYTHONFILES))
PYTHONFILES := $(filter-out eyepieceimageviewer.py, $(PYTHONFILES))
PYTHONFILES := $(filter-out eyepieceviewdialog.py, $(PYTHONFILES))
PYTHONFILES := $(filter-out mainwindow.py, $(PYTHONFILES))
PYTHONFILES := $(filter-out main.py, $(PYTHONFILES))
PYTHONFILES := $(filter-out uistate.py, $(PYTHONFILES))

ui:
	pyuic5 main.ui -o ui.py
	pyuic5 eyepieceview.ui -o ui_eyepieceview.py

lint: $(patsubst %.py,%.pylint,$(PYTHONFILES))

%.pylint:
	$(PYLINT) $(PYLINTFLAGS) $*.py
