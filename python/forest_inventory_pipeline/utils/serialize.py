# MIT License
#
# Copyright (c) 2024 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from json import JSONEncoder
from pathlib import Path


class PathEncoder(JSONEncoder):
    """
    handle pathlib.Path as well.

    check json.JSONEncoder docs for normal argument details.
    Pass any specific params here on to json.dump or json.dumps, the kwargs
    get used for Encoder initialization

    usage: json.dump[s](obj, [fp], cls=PathEncoder, [relative_to=Path(".")])

    :param relative_to: on dump, the written paths can be made relative
    to this path.
    """

    def __init__(self, relative_to: Path | None = None, *args, **kwargs):
        self.relative_to = relative_to
        super().__init__(*args, **kwargs)

    def default(self, o):
        """
        has to return the object itself and not call super.default(o).
        super.default raises an error
        """
        if isinstance(o, Path):
            if self.relative_to is not None:
                return o.relative_to(self.relative_to).as_posix()
            return o.as_posix()
        return super.default(o)
