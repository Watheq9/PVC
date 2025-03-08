import os, sys
sys.path.insert(0, os.getcwd())
from helper import utils
import toml, json
import argparse
import sys
import os, io
import time
import pandas as pd
import traceback
from typing import Any, Dict, List, Literal, Optional, TextIO, Tuple, Union
from enum import Enum, auto


JSONReturnType = Union[Dict[str, Any], List[Any], str, float, int, bool, None]


class ContextValues(Enum):
    OBJECT_KEY = auto()
    OBJECT_VALUE = auto()
    ARRAY = auto()


class JsonContext:
    def __init__(self) -> None:
        self.context: List[ContextValues] = []
        self.current: Optional[ContextValues] = None
        self.empty: bool = True

    def set(self, value: ContextValues) -> None:
        """
        Set a new context value.

        Args:
            value (ContextValues): The context value to be added.

        Returns:
            None
        """
        self.context.append(value)
        self.current = value
        self.empty = False

    def reset(self) -> None:
        """
        Remove the most recent context value.

        Returns:
            None
        """
        try:
            self.context.pop()
            self.current = self.context[-1]
        except IndexError:
            self.current = None
            self.empty = True


class StringFileWrapper:
    # This is a trick to simplify the code, transform the filedescriptor handling into a string handling
    def __init__(self, fd: TextIO, CHUNK_LENGTH: int) -> None:
        """
        Initialize the StringFileWrapper with a file descriptor and chunk length.

        Args:
            fd (TextIO): The file descriptor to wrap.
            CHUNK_LENGTH (int): The length of each chunk to read from the file.

        Attributes:
            fd (TextIO): The wrapped file descriptor.
            length (int): The total length of the file content.
            buffers (dict[int, str]): Dictionary to store chunks of file content.
            buffer_length (int): The length of each buffer chunk.
        """
        self.fd = fd
        self.length: int = 0
        # Buffers are 1MB strings that are read from the file
        # and kept in memory to keep reads low
        self.buffers: dict[int, str] = {}
        # CHUNK_LENGTH is in bytes
        if not CHUNK_LENGTH or CHUNK_LENGTH < 2:
            CHUNK_LENGTH = 1_000_000
        self.buffer_length = CHUNK_LENGTH

    def get_buffer(self, index: int) -> str:
        """
        Retrieve or load a buffer chunk from the file.

        Args:
            index (int): The index of the buffer chunk to retrieve.

        Returns:
            str: The buffer chunk at the specified index.
        """
        if self.buffers.get(index) is None:
            self.fd.seek(index * self.buffer_length)
            self.buffers[index] = self.fd.read(self.buffer_length)
            # Save memory by keeping max 2MB buffer chunks and min 2 chunks
            if len(self.buffers) > max(2, 2_000_000 / self.buffer_length):
                oldest_key = next(iter(self.buffers))
                if oldest_key != index:
                    self.buffers.pop(oldest_key)
        return self.buffers[index]

    def __getitem__(self, index: Union[int, slice]) -> str:
        """
        Retrieve a character or a slice of characters from the file.

        Args:
            index (Union[int, slice]): The index or slice of characters to retrieve.

        Returns:
            str: The character(s) at the specified index or slice.
        """
        # The buffer is an array that is seek like a RAM:
        # self.buffers[index]: the row in the array of length 1MB, index is `i` modulo CHUNK_LENGTH
        # self.buffures[index][j]: the column of the row that is `i` remainder CHUNK_LENGTH
        if isinstance(index, slice):
            buffer_index = index.start // self.buffer_length
            buffer_end = index.stop // self.buffer_length
            if buffer_index == buffer_end:
                return self.get_buffer(buffer_index)[
                    index.start % self.buffer_length : index.stop % self.buffer_length
                ]
            else:
                start_slice = self.get_buffer(buffer_index)[
                    index.start % self.buffer_length :
                ]
                end_slice = self.get_buffer(buffer_end)[
                    : index.stop % self.buffer_length
                ]
                middle_slices = [
                    self.get_buffer(i) for i in range(buffer_index + 1, buffer_end)
                ]
                return start_slice + "".join(middle_slices) + end_slice
        else:
            buffer_index = index // self.buffer_length
            return self.get_buffer(buffer_index)[index % self.buffer_length]

    def __len__(self) -> int:
        """
        Get the total length of the file.

        Returns:
            int: The total number of characters in the file.
        """
        if self.length < 1:
            current_position = self.fd.tell()
            self.fd.seek(0, os.SEEK_END)
            self.length = self.fd.tell()
            self.fd.seek(current_position)
        return self.length

    def __setitem__(self, index: Union[int, slice], value: str) -> None:
        """
        Set a character or a slice of characters in the file.

        Args:
            index (slice): The slice of characters to set.
            value (str): The value to set at the specified index or slice.
        """
        if isinstance(index, slice):
            start = index.start or 0
        else:
            start = index or 0

        if start < 0:
            start += len(self)

        current_position = self.fd.tell()
        self.fd.seek(start)
        self.fd.write(value)
        self.fd.seek(current_position)


class JSONParser:
    # Constants
    STRING_DELIMITERS = ['"', "'", "“", "”"]

    def __init__(
        self,
        json_str: Union[str, StringFileWrapper],
        json_fd: Optional[TextIO],
        logging: Optional[bool],
        json_fd_chunk_length: int = 0,
    ) -> None:
        # The string to parse
        self.json_str: Union[str, StringFileWrapper] = json_str
        # Alternatively, the file description with a json file in it
        if json_fd:
            # This is a trick we do to treat the file wrapper as an array
            self.json_str = StringFileWrapper(json_fd, json_fd_chunk_length)
        # Index is our iterator that will keep track of which character we are looking at right now
        self.index: int = 0
        # This is used in the object member parsing to manage the special cases of missing quotes in key or value
        self.context = JsonContext()
        # Use this to log the activity, but only if logging is active

        # This is a trick but a beatiful one. We call self.log in the code over and over even if it's not needed.
        # We could add a guard in the code for each call but that would make this code unreadable, so here's this neat trick
        # Replace self.log with a noop
        self.logging = logging
        if logging:
            self.logger: List[Dict[str, str]] = []
            self.log = self._log
        else:
            # No-op
            self.log = lambda *args, **kwargs: None

    def parse(
        self,
    ) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
        json = self.parse_json()
        if self.index < len(self.json_str):
            self.log(
                "The parser returned early, checking if there's more json elements",
            )
            json = [json]
            last_index = self.index
            while self.index < len(self.json_str):
                j = self.parse_json()
                if j != "":
                    json.append(j)
                if self.index == last_index:
                    self.index += 1
                last_index = self.index
            # If nothing extra was found, don't return an array
            if len(json) == 1:
                self.log(
                    "There were no more elements, returning the element without the array",
                )
                json = json[0]
        if self.logging:
            return json, self.logger
        else:
            return json

    def parse_json(
        self,
    ) -> JSONReturnType:
        while True:
            char = self.get_char_at()
            # False means that we are at the end of the string provided
            if char is False:
                return ""
            # <object> starts with '{'
            elif char == "{":
                self.index += 1
                return self.parse_object()
            # <array> starts with '['
            elif char == "[":
                self.index += 1
                return self.parse_array()
            # there can be an edge case in which a key is empty and at the end of an object
            # like "key": }. We return an empty string here to close the object properly
            elif self.context.current == ContextValues.OBJECT_VALUE and char == "}":
                self.log(
                    "At the end of an object we found a key with missing value, skipping",
                )
                return ""
            # <string> starts with a quote
            elif not self.context.empty and (
                char in self.STRING_DELIMITERS or char.isalpha()
            ):
                return self.parse_string()
            # <number> starts with [0-9] or minus
            elif not self.context.empty and (
                char.isdigit() or char == "-" or char == "."
            ):
                return self.parse_number()
            # If everything else fails, we just ignore and move on
            else:
                self.index += 1

    def parse_object(self) -> Dict[str, JSONReturnType]:
        # <object> ::= '{' [ <member> *(', ' <member>) ] '}' ; A sequence of 'members'
        obj = {}
        # Stop when you either find the closing parentheses or you have iterated over the entire string
        while (self.get_char_at() or "}") != "}":
            # This is what we expect to find:
            # <member> ::= <string> ': ' <json>

            # Skip filler whitespaces
            self.skip_whitespaces_at()

            # Sometimes LLMs do weird things, if we find a ":" so early, we'll change it to "," and move on
            if (self.get_char_at() or "") == ":":
                self.log(
                    "While parsing an object we found a : before a key, ignoring",
                )
                self.index += 1

            # We are now searching for they string key
            # Context is used in the string parser to manage the lack of quotes
            self.context.set(ContextValues.OBJECT_KEY)

            self.skip_whitespaces_at()

            # Save this index in case we need find a duplicate key
            rollback_index = self.index

            # <member> starts with a <string>
            key = ""
            while self.get_char_at():
                # The rollback index needs to be updated here in case the key is empty
                rollback_index = self.index
                key = str(self.parse_string())

                if key != "" or (key == "" and self.get_char_at() == ":"):
                    # If the string is empty but there is a object divider, we are done here
                    break
            if ContextValues.ARRAY in self.context.context and key in obj:
                self.log(
                    "While parsing an object we found a duplicate key, closing the object here and rolling back the index",
                )
                self.index = rollback_index - 1
                # add an opening curly brace to make this work
                self.json_str = (
                    self.json_str[: self.index + 1]
                    + "{"
                    + self.json_str[self.index + 1 :]
                )
                break

            # Skip filler whitespaces
            self.skip_whitespaces_at()

            # We reached the end here
            if (self.get_char_at() or "}") == "}":
                continue

            self.skip_whitespaces_at()

            # An extreme case of missing ":" after a key
            if (self.get_char_at() or "") != ":":
                self.log(
                    "While parsing an object we missed a : after a key",
                )

            self.index += 1
            self.context.reset()
            self.context.set(ContextValues.OBJECT_VALUE)
            # The value can be any valid json
            value = self.parse_json()

            # Reset context since our job is done
            self.context.reset()
            obj[key] = value

            if (self.get_char_at() or "") in [",", "'", '"']:
                self.index += 1

            # Remove trailing spaces
            self.skip_whitespaces_at()

        self.index += 1
        return obj

    def parse_array(self) -> List[JSONReturnType]:
        # <array> ::= '[' [ <json> *(', ' <json>) ] ']' ; A sequence of JSON values separated by commas
        arr = []
        self.context.set(ContextValues.ARRAY)
        # Stop when you either find the closing parentheses or you have iterated over the entire string
        char = self.get_char_at()
        while char and char not in ["]", "}"]:
            self.skip_whitespaces_at()
            value = self.parse_json()

            # It is possible that parse_json() returns nothing valid, so we stop
            if value == "":
                break

            if value == "..." and self.get_char_at(-1) == ".":
                self.log(
                    "While parsing an array, found a stray '...'; ignoring it",
                )
            else:
                arr.append(value)

            # skip over whitespace after a value but before closing ]
            char = self.get_char_at()
            while char and (char.isspace() or char == ","):
                self.index += 1
                char = self.get_char_at()

        # Especially at the end of an LLM generated json you might miss the last "]"
        char = self.get_char_at()
        if char and char != "]":
            self.log(
                "While parsing an array we missed the closing ], adding it back",
            )
            self.index -= 1

        self.index += 1
        self.context.reset()
        return arr

    def parse_string(self) -> Union[str, bool, None]:
        # <string> is a string of valid characters enclosed in quotes
        # i.e. { name: "John" }
        # Somehow all weird cases in an invalid JSON happen to be resolved in this function, so be careful here

        # Flag to manage corner cases related to missing starting quote
        missing_quotes = False
        doubled_quotes = False
        lstring_delimiter = rstring_delimiter = '"'

        char = self.get_char_at()
        # A valid string can only start with a valid quote or, in our case, with a literal
        while char and char not in self.STRING_DELIMITERS and not char.isalnum():
            self.index += 1
            char = self.get_char_at()

        if not char:
            # This is an empty string
            return ""

        # Ensuring we use the right delimiter
        if char == "'":
            lstring_delimiter = rstring_delimiter = "'"
        elif char == "“":
            lstring_delimiter = "“"
            rstring_delimiter = "”"
        elif char.isalnum():
            # This could be a <boolean> and not a string. Because (T)rue or (F)alse or (N)ull are valid
            # But remember, object keys are only of type string
            if (
                char.lower() in ["t", "f", "n"]
                and self.context.current != ContextValues.OBJECT_KEY
            ):
                value = self.parse_boolean_or_null()
                if value != "":
                    return value
            self.log(
                "While parsing a string, we found a literal instead of a quote",
            )
            self.log(
                "While parsing a string, we found no starting quote. Will add the quote back",
            )
            missing_quotes = True

        if not missing_quotes:
            self.index += 1

        self.skip_whitespaces_at()
        # There is sometimes a weird case of doubled quotes, we manage this also later in the while loop
        if self.get_char_at() in self.STRING_DELIMITERS:
            # If the next character is the same type of quote, then we manage it as double quotes
            if self.get_char_at() == lstring_delimiter:
                # If it's an empty key, this was easy
                if (
                    self.context.current == ContextValues.OBJECT_KEY
                    and self.get_char_at(1) == ":"
                ):
                    self.index += 1
                    return ""
                if self.get_char_at(1) == lstring_delimiter:
                    # There's something fishy about this, we found doubled quotes and then again quotes
                    self.log(
                        "While parsing a string, we found a doubled quote and then a quote again, ignoring it",
                    )
                    return ""
                # Find the next delimiter
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                # Now check that the next character is also a delimiter to ensure that we have "".....""
                # In that case we ignore this rstring delimiter
                if next_c and (self.get_char_at(i + 1) or "") == rstring_delimiter:
                    self.log(
                        "While parsing a string, we found a valid starting doubled quote",
                    )
                    doubled_quotes = True
                    self.index += 1
                else:
                    # Ok this is not a doubled quote, check if this is an empty string or not
                    i = self.skip_whitespaces_at(idx=1, move_main_index=False)
                    next_c = self.get_char_at(i)
                    if next_c in self.STRING_DELIMITERS + ["{", "["]:
                        # something fishy is going on here
                        self.log(
                            "While parsing a string, we found a doubled quote but also another quote afterwards, ignoring it",
                        )
                        self.index += 1
                        return ""
                    elif next_c not in [",", "]", "}"]:
                        self.log(
                            "While parsing a string, we found a doubled quote but it was a mistake, removing one quote",
                        )
                        self.index += 1
            else:
                # Otherwise we need to do another check before continuing
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                if not next_c:
                    # mmmm that delimiter never appears again, this is a mistake
                    self.log(
                        "While parsing a string, we found a quote but it was a mistake, ignoring it",
                    )
                    return ""

        # Initialize our return value
        string_acc = ""

        # Here things get a bit hairy because a string missing the final quote can also be a key or a value in an object
        # In that case we need to use the ":|,|}" characters as terminators of the string
        # So this will stop if:
        # * It finds a closing quote
        # * It iterated over the entire sequence
        # * If we are fixing missing quotes in an object, when it finds the special terminators
        char = self.get_char_at()
        unmatched_delimiter = False
        while char and char != rstring_delimiter:
            if (
                missing_quotes
                and self.context.current == ContextValues.OBJECT_KEY
                and (char == ":" or char.isspace())
            ):
                self.log(
                    "While parsing a string missing the left delimiter in object key context, we found a :, stopping here",
                )
                break
            if self.context.current == ContextValues.OBJECT_VALUE and char in [
                ",",
                "}",
            ]:
                rstring_delimiter_missing = True
                # check if this is a case in which the closing comma is NOT missing instead
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                if next_c:
                    i += 1
                    # found a delimiter, now we need to check that is followed strictly by a comma or brace
                    # or the string ended
                    i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                    next_c = self.get_char_at(i)
                    if not next_c or next_c in [",", "}"]:
                        rstring_delimiter_missing = False
                    else:
                        # OK but this could still be some garbage at the end of the string
                        # So we need to check if we find a new lstring_delimiter afterwards
                        # If we do, maybe this is a missing delimiter
                        i = self.skip_to_character(character=lstring_delimiter, idx=i)
                        if doubled_quotes:
                            i = self.skip_to_character(
                                character=lstring_delimiter, idx=i
                            )
                        next_c = self.get_char_at(i)
                        if not next_c:
                            rstring_delimiter_missing = False
                        else:
                            # But again, this could just be something a bit stupid like "lorem, "ipsum" sic"
                            # Check if we find a : afterwards (skipping space)
                            i = self.skip_whitespaces_at(
                                idx=i + 1, move_main_index=False
                            )
                            next_c = self.get_char_at(i)
                            if next_c and next_c != ":":
                                rstring_delimiter_missing = False
                else:
                    # There could be a case in which even the next key:value is missing delimeters
                    # because it might be a systemic issue with the output
                    # So let's check if we can find a : in the string instead
                    i = self.skip_to_character(character=":", idx=1)
                    next_c = self.get_char_at(i)
                    if next_c:
                        # OK then this is a systemic issue with the output
                        break
                    else:
                        # skip any whitespace first
                        i = self.skip_whitespaces_at(idx=1, move_main_index=False)
                        # We couldn't find any rstring_delimeter before the end of the string
                        # check if this is the last string of an object and therefore we can keep going
                        # make an exception if this is the last char before the closing brace
                        j = self.skip_to_character(character="}", idx=i)
                        if j - i > 1:
                            # Ok it's not right after the comma
                            # Let's ignore
                            rstring_delimiter_missing = False
                        # Check that j was not out of bound
                        elif self.get_char_at(j):
                            # Check for an unmatched opening brace in string_acc
                            for c in reversed(string_acc):
                                if c == "{":
                                    # Ok then this is part of the string
                                    rstring_delimiter_missing = False
                                    break
                                elif c == "}":
                                    break
                if rstring_delimiter_missing:
                    self.log(
                        "While parsing a string missing the left delimiter in object value context, we found a , or } and we couldn't determine that a right delimiter was present. Stopping here",
                    )
                    break
            if char == "]" and ContextValues.ARRAY in self.context.context:
                # We found the end of an array and we are in array context
                # So let's check if we find a rstring_delimiter forward otherwise end early
                i = self.skip_to_character(rstring_delimiter)
                if not self.get_char_at(i):
                    # No delimiter found
                    break
            string_acc += char
            self.index += 1
            char = self.get_char_at()
            if char and len(string_acc) > 0 and string_acc[-1] == "\\":
                # This is a special case, if people use real strings this might happen
                self.log("Found a stray escape sequence, normalizing it")
                if char in [rstring_delimiter, "t", "n", "r", "b", "\\"]:
                    string_acc = string_acc[:-1]
                    escape_seqs = {"t": "\t", "n": "\n", "r": "\r", "b": "\b"}
                    string_acc += escape_seqs.get(char, char) or char
                    self.index += 1
                    char = self.get_char_at()
            # If we are in object key context and we find a colon, it could be a missing right quote
            if (
                char == ":"
                and not missing_quotes
                and self.context.current == ContextValues.OBJECT_KEY
            ):
                # Ok now we need to check if this is followed by a value like "..."
                i = self.skip_to_character(character=lstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                if next_c:
                    i += 1
                    # found the first delimiter
                    i = self.skip_to_character(character=rstring_delimiter, idx=i)
                    next_c = self.get_char_at(i)
                    if next_c:
                        # found a second delimiter
                        i += 1
                        # Skip spaces
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)
                        if next_c and next_c in [",", "}"]:
                            # Ok then this is a missing right quote
                            self.log(
                                "While parsing a string missing the right delimiter in object key context, we found a :, stopping here",
                            )
                            break
                else:
                    # The string ended without finding a lstring_delimiter, I will assume this is a missing right quote
                    self.log(
                        "While parsing a string missing the right delimiter in object key context, we found a :, stopping here",
                    )
                    break
            # ChatGPT sometimes forget to quote stuff in html tags or markdown, so we do this whole thing here
            if char == rstring_delimiter:
                # Special case here, in case of double quotes one after another
                if doubled_quotes and self.get_char_at(1) == rstring_delimiter:
                    self.log(
                        "While parsing a string, we found a doubled quote, ignoring it"
                    )
                    self.index += 1
                elif (
                    missing_quotes
                    and self.context.current == ContextValues.OBJECT_VALUE
                ):
                    # In case of missing starting quote I need to check if the delimeter is the end or the beginning of a key
                    i = 1
                    next_c = self.get_char_at(i)
                    while next_c and next_c not in [
                        rstring_delimiter,
                        lstring_delimiter,
                    ]:
                        i += 1
                        next_c = self.get_char_at(i)
                    if next_c:
                        # We found a quote, now let's make sure there's a ":" following
                        i += 1
                        # found a delimiter, now we need to check that is followed strictly by a comma or brace
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)
                        if next_c and next_c == ":":
                            # Reset the cursor
                            self.index -= 1
                            char = self.get_char_at()
                            self.log(
                                "In a string with missing quotes and object value context, I found a delimeter but it turns out it was the beginning on the next key. Stopping here.",
                            )
                            break
                elif unmatched_delimiter:
                    unmatched_delimiter = False
                    string_acc += str(char)
                    self.index += 1
                    char = self.get_char_at()
                else:
                    # Check if eventually there is a rstring delimiter, otherwise we bail
                    i = 1
                    next_c = self.get_char_at(i)
                    check_comma_in_object_value = True
                    while next_c and next_c not in [
                        rstring_delimiter,
                        lstring_delimiter,
                    ]:
                        # This is a bit of a weird workaround, essentially in object_value context we don't always break on commas
                        # This is because the routine after will make sure to correct any bad guess and this solves a corner case
                        if check_comma_in_object_value and next_c.isalpha():
                            check_comma_in_object_value = False
                        # If we are in an object context, let's check for the right delimiters
                        if (
                            (
                                ContextValues.OBJECT_KEY in self.context.context
                                and next_c in [":", "}"]
                            )
                            or (
                                ContextValues.OBJECT_VALUE in self.context.context
                                and next_c == "}"
                            )
                            or (
                                ContextValues.ARRAY in self.context.context
                                and next_c in ["]", ","]
                            )
                            or (
                                check_comma_in_object_value
                                and self.context.current == ContextValues.OBJECT_VALUE
                                and next_c == ","
                            )
                        ):
                            break
                        i += 1
                        next_c = self.get_char_at(i)
                    # If we stopped for a comma in object_value context, let's check if find a "} at the end of the string
                    if (
                        next_c == ","
                        and self.context.current == ContextValues.OBJECT_VALUE
                    ):
                        i += 1
                        i = self.skip_to_character(character=rstring_delimiter, idx=i)
                        next_c = self.get_char_at(i)
                        # Ok now I found a delimiter, let's skip whitespaces and see if next we find a }
                        i += 1
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)
                        if next_c == "}":
                            # OK this is valid then
                            self.log(
                                "While parsing a string, we misplaced a quote that would have closed the string but has a different meaning here since this is the last element of the object, ignoring it",
                            )
                            unmatched_delimiter = not unmatched_delimiter
                            string_acc += str(char)
                            self.index += 1
                            char = self.get_char_at()
                    elif (
                        next_c == rstring_delimiter and self.get_char_at(i - 1) != "\\"
                    ):
                        if self.context.current == ContextValues.OBJECT_VALUE:
                            # But this might not be it! This could be just a missing comma
                            # We found a delimiter and we need to check if this is a key
                            # so find a rstring_delimiter and a colon after
                            i = self.skip_to_character(
                                character=rstring_delimiter, idx=i + 1
                            )
                            i += 1
                            next_c = self.get_char_at(i)
                            while next_c and next_c != ":":
                                if next_c == "," or (
                                    next_c == rstring_delimiter
                                    and self.get_char_at(i - 1) != "\\"
                                ):
                                    break
                                i += 1
                                next_c = self.get_char_at(i)
                            # Only if we fail to find a ':' then we know this is misplaced quote
                            if next_c != ":":
                                self.log(
                                    "While parsing a string, we a misplaced quote that would have closed the string but has a different meaning here, ignoring it",
                                )
                                unmatched_delimiter = not unmatched_delimiter
                                string_acc += str(char)
                                self.index += 1
                                char = self.get_char_at()
                        elif self.context.current == ContextValues.ARRAY:
                            # In array context this could be something like "lorem "ipsum" sic"
                            # So let's check if we find a rstring_delimiter forward otherwise end early
                            i = self.skip_to_character(rstring_delimiter, idx=i + 1)
                            next_c = self.get_char_at(i)
                            if next_c and next_c == rstring_delimiter:
                                # Ok now if I find a comma or a closing ], that can be have also an optional rstring_delimiter before them
                                # We can consider this a misplaced quote
                                i += 1
                                i = self.skip_whitespaces_at(
                                    idx=i, move_main_index=False
                                )
                                next_c = self.get_char_at(i)
                                if next_c and next_c in [",", "]"]:
                                    self.log(
                                        "While parsing a string, we a misplaced quote that would have closed the string but has a different meaning here, ignoring it",
                                    )
                                    unmatched_delimiter = not unmatched_delimiter
                                    string_acc += str(char)
                                    self.index += 1
                                    char = self.get_char_at()

        if (
            char
            and missing_quotes
            and self.context.current == ContextValues.OBJECT_KEY
            and char.isspace()
        ):
            self.log(
                "While parsing a string, handling an extreme corner case in which the LLM added a comment instead of valid string, invalidate the string and return an empty value",
            )
            self.skip_whitespaces_at()
            if self.get_char_at() not in [":", ","]:
                return ""

        # A fallout of the previous special case in the while loop,
        # we need to update the index only if we had a closing quote
        if char != rstring_delimiter:
            self.log(
                "While parsing a string, we missed the closing quote, ignoring",
            )
        else:
            self.index += 1

        return string_acc.rstrip()

    def parse_number(self) -> Union[float, int, str, JSONReturnType]:
        # <number> is a valid real number expressed in one of a number of given formats
        number_str = ""
        number_chars = set("0123456789-.eE/,")
        char = self.get_char_at()
        is_array = self.context.current == ContextValues.ARRAY
        while char and char in number_chars and (char != "," or not is_array):
            number_str += char
            self.index += 1
            char = self.get_char_at()
        if len(number_str) > 1 and number_str[-1] in "-eE/,":
            # The number ends with a non valid character for a number/currency, rolling back one
            number_str = number_str[:-1]
            self.index -= 1
        try:
            if "," in number_str:
                return str(number_str)
            if "." in number_str or "e" in number_str or "E" in number_str:
                return float(number_str)
            elif number_str == "-":
                # If there is a stray "-" this will throw an exception, throw away this character
                return self.parse_json()
            else:
                return int(number_str)
        except ValueError:
            return number_str

    def parse_boolean_or_null(self) -> Union[bool, str, None]:
        # <boolean> is one of the literal strings 'true', 'false', or 'null' (unquoted)
        starting_index = self.index
        char = (self.get_char_at() or "").lower()
        value: Optional[Tuple[str, Optional[bool]]]
        if char == "t":
            value = ("true", True)
        elif char == "f":
            value = ("false", False)
        elif char == "n":
            value = ("null", None)

        if value:
            i = 0
            while char and i < len(value[0]) and char == value[0][i]:
                i += 1
                self.index += 1
                char = (self.get_char_at() or "").lower()
            if i == len(value[0]):
                return value[1]

        # If nothing works reset the index before returning
        self.index = starting_index
        return ""

    def get_char_at(self, count: int = 0) -> Union[str, Literal[False]]:
        # Why not use something simpler? Because try/except in python is a faster alternative to an "if" statement that is often True
        try:
            return self.json_str[self.index + count]
        except IndexError:
            return False

    def skip_whitespaces_at(self, idx: int = 0, move_main_index=True) -> int:
        """
        This function quickly iterates on whitespaces, syntactic sugar to make the code more concise
        """
        try:
            char = self.json_str[self.index + idx]
        except IndexError:
            return idx
        while char.isspace():
            if move_main_index:
                self.index += 1
            else:
                idx += 1
            try:
                char = self.json_str[self.index + idx]
            except IndexError:
                return idx
        return idx

    def skip_to_character(self, character: str, idx: int = 0) -> int:
        """
        This function quickly iterates to find a character, syntactic sugar to make the code more concise
        """
        try:
            char = self.json_str[self.index + idx]
        except IndexError:
            return idx
        while char != character:
            idx += 1
            try:
                char = self.json_str[self.index + idx]
            except IndexError:
                return idx
        if self.index + idx > 0 and self.json_str[self.index + idx - 1] == "\\":
            # Ah this is an escaped character, try again
            return self.skip_to_character(character=character, idx=idx + 1)
        return idx


def repair_json(
    json_str: str = "",
    return_objects: bool = False,
    skip_json_loads: bool = False,
    logging: bool = False,
    json_fd: Optional[TextIO] = None,
    ensure_ascii: bool = True,
    chunk_length: int = 0,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    Given a json formatted string, it will try to decode it and, if it fails, it will try to fix it.

    Args:
        json_str (str, optional): The JSON string to repair. Defaults to an empty string.
        return_objects (bool, optional): If True, return the decoded data structure. Defaults to False.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.
        json_fd (Optional[TextIO], optional): File descriptor for JSON input. Do not use! Use `from_file` or `load` instead. Defaults to None.
        ensure_ascii (bool, optional): Set to False to avoid converting non-latin characters to ascii (for example when using chinese characters). Defaults to True. Ignored if `skip_json_loads` is True.
        chunk_length (int, optional): Size in bytes of the file chunks to read at once. Ignored if `json_fd` is None. Do not use! Use `from_file` or `load` instead. Defaults to 1MB.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON or a tuple with the repaired JSON and repair log.
    """
    parser = JSONParser(json_str, json_fd, logging, chunk_length)
    if skip_json_loads:
        parsed_json = parser.parse()
    else:
        try:
            if json_fd:
                parsed_json = json.load(json_fd)
            else:
                parsed_json = json.loads(json_str)
        except json.JSONDecodeError:
            parsed_json = parser.parse()
    # It's useful to return the actual object instead of the json string,
    # it allows this lib to be a replacement of the json library
    if return_objects or logging:
        return parsed_json
    return json.dumps(parsed_json, ensure_ascii=ensure_ascii)


def loads(
    json_str: str,
    skip_json_loads: bool = False,
    logging: bool = False,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    This function works like `json.loads()` except that it will fix your JSON in the process.
    It is a wrapper around the `repair_json()` function with `return_objects=True`.

    Args:
        json_str (str): The JSON string to load and repair.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON object or a tuple with the repaired JSON object and repair log.
    """
    return repair_json(
        json_str=json_str,
        return_objects=True,
        skip_json_loads=skip_json_loads,
        logging=logging,
    )


def load(
    fd: TextIO,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    This function works like `json.load()` except that it will fix your JSON in the process.
    It is a wrapper around the `repair_json()` function with `json_fd=fd` and `return_objects=True`.

    Args:
        fd (TextIO): File descriptor for JSON input.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.
        chunk_length (int, optional): Size in bytes of the file chunks to read at once. Defaults to 1MB.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON object or a tuple with the repaired JSON object and repair log.
    """
    return repair_json(
        json_fd=fd,
        chunk_length=chunk_length,
        return_objects=True,
        skip_json_loads=skip_json_loads,
        logging=logging,
    )


def from_file(
    filename: str,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    This function is a wrapper around `load()` so you can pass the filename as string

    Args:
        filename (str): The name of the file containing JSON data to load and repair.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.
        chunk_length (int, optional): Size in bytes of the file chunks to read at once. Defaults to 1MB.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON object or a tuple with the repaired JSON object and repair log.
    """
    with open(filename) as fd:
        jsonobj = load(
            fd=fd,
            skip_json_loads=skip_json_loads,
            logging=logging,
            chunk_length=chunk_length,
        )

    return jsonobj


def cli(inline_args: Optional[List[str]] = None) -> int:
    """
    Command-line interface for repairing and parsing JSON files.

    Args:
        inline_args (Optional[List[str]]): List of command-line arguments for testing purposes. Defaults to None.
            - filename (str): The JSON file to repair
            - -i, --inline (bool): Replace the file inline instead of returning the output to stdout.
            - -o, --output TARGET (str): If specified, the output will be written to TARGET filename instead of stdout.
            - --ensure_ascii (bool): Pass ensure_ascii=True to json.dumps(). Will pass False otherwise.
            - --indent INDENT (int): Number of spaces for indentation (Default 2).

    Returns:
        int: Exit code of the CLI operation.

    Raises:
        Exception: Any exception that occurs during file processing.

    Example:
        >>> cli(['example.json', '--indent', '4'])
    """
    parser = argparse.ArgumentParser(description="Repair and parse JSON files.")
    parser.add_argument("filename", help="The JSON file to repair")
    parser.add_argument(
        "-i",
        "--inline",
        action="store_true",
        help="Replace the file inline instead of returning the output to stdout",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="TARGET",
        help="If specified, the output will be written to TARGET filename instead of stdout",
    )
    parser.add_argument(
        "--ensure_ascii",
        action="store_true",
        help="Pass ensure_ascii=True to json.dumps()",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Number of spaces for indentation (Default 2)",
    )

    if inline_args is None:  # pragma: no cover
        args = parser.parse_args()
    else:
        args = parser.parse_args(
            inline_args
        )  # This is needed so this function is testable

    if args.inline and args.output:  # pragma: no cover
        print("Error: You cannot pass both --inline and --output", file=sys.stderr)
        sys.exit(1)

    ensure_ascii = False
    if args.ensure_ascii:
        ensure_ascii = True

    try:
        result = from_file(args.filename)

        if args.inline or args.output:
            with open(args.output or args.filename, mode="w") as fd:
                json.dump(result, fd, indent=args.indent, ensure_ascii=ensure_ascii)
        else:
            print(json.dumps(result, indent=args.indent, ensure_ascii=ensure_ascii))
    except Exception as e:  # pragma: no cover
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

    return 0  # Success


def read_missing(file):
    dic = ''
    with open(file,'r') as f:
        for i in f.readlines():
            dic=i #string
    dic = eval(dic) 
    return dic


def extract_efficiency_data(text_to_parse, json_data):
  
    VALA = "load time = "
    VALB = "prompt eval time = "
    VALC = "total time = "

    lines = text_to_parse.split("\n")
    print(type(json_data))
    for line in lines:
        print(line)
        if VALA in line:
            fields = line.strip().split()
            print(fields[4])
            json_data['load_time_ms'] = float(fields[4])
            # json_data.update({'load_time_ms':  float(fields[4])})
        elif VALB in line:
            fields = line.strip().split()
            json_data['input_tokens'] = int(fields[8])
        elif VALC in line:
            fields = line.strip().split()
            json_data['total_time_ms'] = float(fields[4])
            json_data['total_tokens'] = int(fields[7])

    json_data['output_tokens'] = json_data['total_tokens'] - json_data['input_tokens']
    json_data['run_time_ms'] = round((json_data['total_time_ms'] - json_data['load_time_ms']),2)
    return json_data



def run_llm(model_name, input, prompt_template, result, missing_file, few_shots_file, logger, format='tsv'):
    from llama_cpp import Llama

    prompt_template = toml.load(prompt_template)
    if input.endswith('jsonl'):
        df_pairs  = utils.read_jsonl(input)
    elif input.endswith('tsv'): 
        df_pairs  = pd.read_csv(input, sep='\t',) # names=['qid', 'docno', 'q_description', 'doc_text'])
    else:
        raise Exception("Unsupported input type")

    missing_pairs = None
    if missing_file != "":
        missing_pairs = read_missing(missing_file)

    judged_pairs_dict = {}
    if os.path.exists(result): # some pairs already judged and we need to exclude them
        df_res = pd.read_json(result, lines=True)
        for row in df_res.itertuples():
            judged_pairs_dict[(str(row.qid), str(row.docno))] = 1
        print(f"length of already judged pairs {len(judged_pairs_dict)}")

    llm = Llama(model_path=model_name, n_ctx=2048, n_gpu_layers=64, chat_format="llama-2")

    
    cnt = 0
    missing = {}
    # utils.write_line(result, "", mode='w')
    for row in df_pairs.itertuples():
        try:   
            # Create StringIO objects to capture output
            # stdout_capture = io.StringIO()
            # stderr_capture = io.StringIO()
            # Redirect stdout and stderr
            # sys.stdout = stdout_capture
            # sys.stderr = stderr_capture

            qid = str(row.qid)
            docno = str(row.docno)
            # query = row.q_text
            query_description = row.q_description
            segment = row.doc_text

            if segment == "nan" or segment == "" or segment == r"\s+":
                print(f"Pair (qid={qid}, docno={docno}) has empty segment with value {segment}")
                continue
            
            if (qid, docno) in judged_pairs_dict:
                print(f"Pair (qid={qid}, docno={docno}) is already judged")
                continue

            if missing_pairs is not None:
                in_missing= missing_pairs.get((qid, docno), 0)
                if in_missing == 0:
                    continue
            
            cnt +=1 
            # user_prompt = prompt_template['user'].format(query=query, query_description=query_description, segment=segment)
            user_prompt = prompt_template['user'].format(query_description=query_description, segment=segment)
            system_prompt = prompt_template["system"]
            # add few shots to the prompt
            if few_shots_file != "":
                examples = utils.read_json(few_shots_file)
                user_prompt = user_prompt.format(examples=examples)
            # num_tokens = gpt_judge.num_tokens_from_string(system_prompt, model=model) + gpt_judge.num_tokens_from_string(user_prompt, model=model)
            # logger.info(f"For the pair of qid = {qid}, docno ={docno}, the number of input tokens (system & user prompts) is {num_tokens}")

            start_time = time.time()
            response = llm.create_chat_completion(
                            messages = [
                            {"role": "system", "content": system_prompt},
                            { "role": "user", "content": user_prompt}
                            ],
                        response_format={ "type": "json_object", },
                        temperature=0.0,
                        top_p=1,
                        frequency_penalty=0.5,
                        presence_penalty=0,
                        )

            jstr = repair_json(json_str=str(response))
            obj = json.loads(jstr)
            output = json.loads(obj["choices"][0]["message"]["content"])
            # print(output)
            # print(obj["choices"][0]["message"]["content"])

            exec_time = round(time.time() - start_time, 3)

            # Reset stdout and stderr to default
            # sys.stdout = sys.__stdout__
            # sys.stderr = sys.__stderr__

            # Get captured output
            # stdout_content = stdout_capture.getvalue()
            # stderr_content = stderr_capture.getvalue()

            if output == "":
                missing.update({(str(qid), str(docno)): 1})
                continue

            # output = extract_efficiency_data(text_to_parse=stderr_content, json_data=json.loads(output))
            line = {"qid": qid, "docno": docno, "exec_time": exec_time}
            line.update(output)
            # logger.info(f"Time for judging the pair of qid = {qid}, docno ={docno} is {exec_time:.3f} seconds and run_time_ms {output['run_time_ms']}")

            line = json.dumps(line) + '\n'

            utils.write_line(result, line, mode='a')

        except Exception as e:
            # print(e.__traceback__())
            logger.error(f"When processing query id qid = {qid}, docno ={docno}, we got the following error: {e}")
            logger.info(obj["choices"][0]["message"]["content"])
            logger.info(traceback.format_exc())
            continue
        
    logger.info(f"missing = {missing}")


def sanity_check(input, prompt_template, result, logger):
    if os.path.exists(input):
        logger.info(f"Input file does exist at {input}")
    else:
        logger.error(f"Input file does NOT exist at {input}")
    
    if os.path.exists(prompt_template):
        logger.info("prompt_template file does exist")
    else:
        logger.error(f"prompt_template file does NOT exist at {prompt_template}")

    if os.path.exists(result):
        df_res = pd.read_json(result, lines=True)
        logger.info(f"Result file does exist and number of pairs already judged are {len(df_res)}")
    else:
        logger.info(f"Result file at {result} does NOT exist and trying to add one empty string")
        utils.write_line(result, "", mode='w')
        logger.info(f"Written successuflly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Name of LLM model")
    parser.add_argument("--input", required=True, type=str, help="File containing the query-document pairs to judge with LLM")
    parser.add_argument("--prompt_template", required=True, type=str, help="prompt template")
    parser.add_argument("--result", required=True, type=str, help="File to save the LLM output")
    parser.add_argument("--log_file", required=True, type=str, help="File to save the log")
    parser.add_argument("--max_tokens", required=False, type=int, default=150, help="max_tokens")
    parser.add_argument("--missing_file", required=False, type=str, default="", help="file to missing pairs to re-evaluate")
    parser.add_argument("--few_shots_file", required=False, type=str, default="", help="file contains few shot examples")
    parser.add_argument("--just_check", required=False, action="store_true", default=False, help="Flag to check if all input and output files are sound and then exit")
    args = parser.parse_args()
    # parser.add_argument("--prompt_type", required=False, default="user", type=str, help="The type of the template: can be system or user")


    # model = "gpt-4o-2024-11-20"
    model = args.model
    input = args.input
    prompt_template = args.prompt_template
    result = args.result
    log_file = args.log_file
    # output_mode = args.output_mode
    # prompt_type = args.prompt_type
    max_tokens = args.max_tokens
    missing_file = args.missing_file
    few_shots_file = args.few_shots_file
    just_check = args.just_check
    logger = utils.get_logger(log_file=log_file)

    sanity_check(input, prompt_template, result, logger)
    if just_check:
        return
        
    run_llm(model, input, prompt_template, result, missing_file, few_shots_file, logger)
    

if __name__ == "__main__":
    main()

