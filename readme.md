# Practical 1 - Console App For Data Analysis

240030614

## Project Outline

* Typer for console interface (should be more modern than tkinter)

Console choices:

* Typer (parse args) + Rich (for output formatting)
* maybe textual

### textual

in the first console window, run: `textual console`, then in the second one:

```
textual run --dev my_app.py
```

## Preliminary Research - How to start this?

### functionality - part 1

* read_csv
* store dataset (which methods provided?)
* search by:
  * (local authority / time_period)
  * (school_type / absences / time_period)
  * unauthorised absences in certain year  by region name / local auth name

### functionality - part 2

* compare local auths (use search beforehand)
* explore performances etc

### functionality - part 3

* explore links between data (how?)

## infos about sparks

### spark sessions

master() allows: `local[x]` with x being the number of partitions (ideally = cores) or `spark://master:7077` for standalone

### visualise spark df in matplotlib

* how is this accomplished?