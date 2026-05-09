# Refactoring

- replace the statistics logging with accumilation of data, then frequently generate/update/append to graphs from those statistics.
    - lib: plotters
    - can i make the graph update in realtime like less +F main.log?

- log the selfplay gamnes into a dedicated .pgn file. mate in 1s should be level debug, mate in 2s should be level verbose, anthing else is level info... or something like that

- refine and cleanup the pipeline.
