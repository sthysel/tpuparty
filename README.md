# TPUParty (Version 0.0.1)

A set of tools and toys to work with the Google TPU

# Usage

``` zsh
Usage: tpuparty [OPTIONS]

Options:
  --modeldir TEXT         Directory containing the model weight and label
                          files  [default: ~/models/coco/]
  -f, --video-in TEXT     Video inout file  [default: test.mkv]
  -c, --confidence FLOAT  Confidence threshold for object inference  [default:
                          0.1]
  --version               Show the version and exit.
  --help                  Show this message and exit.
```


# Install

At this time tensorflow is not yet available for Python 3.8, so use the next best thing

```zsh
$ mkvirtualenv -p python3.7 tpuparty 
$ pip install .
```
