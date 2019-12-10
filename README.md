# TPUParty (Version 0.0.1)

A set of tools and toys to work with the Google TPU

# Usage

``` zsh
Usage: tpuparty [OPTIONS] SOURCE

  Runs inference over source

  Examples:
  $ tpuparty "http://10.0.0.185/axis-cgi/mjpg/video.cgi?&camera=2"
  $ tpuparty 0

Options:
  --modeldir TEXT         Directory containing the model weight and label
                          files  [default: ~/models/coco/]
  -c, --confidence FLOAT  Confidence threshold for object inference  [default: 0.1]
  --fps TEXT              FPS playback for recordings
  --version               Show the version and exit.
  --help                  Show this message and exit.
```

# Models

`tpuparty` expects models to be presented in directories containing at least a
graph.tflite file like so:

```zsh
models
└── coco/
    ├── graph.tflite
    ├── labels.txt
    └── README.md
```

The common COCO trained mobilenet model is included in this repo for
convenience.

Notice the models are sourced from various places and may have their own licences attached.
The licence for this project pertains to the code only. 

# Install

At this time tensorflow is not yet available for Python 3.8, so use the next best thing

```zsh
$ mkvirtualenv -p python3.7 tpuparty 
$ pip install .
```

