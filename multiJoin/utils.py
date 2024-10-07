from typing import Literal, Optional, Generator, Tuple, Dict, List, Set, Any, Union
from collections import defaultdict, deque
import io
import re
import os
import csv
import math
import time
import copy
import string
import random
from datetime import datetime

import torch
from torch.fft import fft, ifft
import numpy
import pandas as pd
from tap import Tap
from tqdm import tqdm

from kwisehash import KWiseHash

import heapq
import sys

camel_to_snake_re = re.compile(r"(?<!^)(?=[A-Z])")
selection_ops_re = re.compile(r"(\>\=?|\<\=?|\<\>|\=|BETWEEN|IN|LIKE|NOT LIKE)")
attribute_re = re.compile(r"(_|[a-zA-Z])(_|\d|[a-zA-Z])*.(_|[a-zA-Z])+")
escaped_backslash_re = re.compile(r"\\\"")

NULL_VALUE = -123456
CARDEST_DIR = "End-to-End-CardEst-Benchmark-master"
IMDB_DIR = "imdb"
CACHE_DIR = ".cache"


# http://en.wikipedia.org/wiki/Mersenne_prime
MERSENNE_PRIME = (1 << 61) - 1

def text_between(input, start, end):
    # getting index of substrings
    idx_start = input.index(start)
    idx_end = input.index(end)

    # length of substring 1 is added to
    # get string from next character
    return input[idx_start + len(start) + 1 : idx_end]

def is_number(n):
    try:
        # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
        float(n)
    except ValueError:
        return False
    return True

def random_string(len = 7) -> str:
    chars = string.ascii_letters + string.digits
    rand_chars = random.choices(chars, k=len)
    rand_str = "".join(rand_chars)
    return rand_str

class Timer(object):
    def __init__(self):
        self.start = time.perf_counter()

    def stop(self):
        return time.perf_counter() - self.start

class SignHash(object):
    fn: KWiseHash

    def __init__(self, *size, k=2) -> None:
        self.fn = KWiseHash(*size, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.sign(items)

class ComposedSigns(object):
    hashes: List[SignHash]

    def __init__(self, *hashes: SignHash) -> None:
        self.hashes = hashes

    def add(self, hash: SignHash) -> None:
        self.hashes.append(hash)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        result = 1

        for hash in self.hashes:
            result *= hash(items)

        return result

class BinHash(object):
    fn: KWiseHash

    def __init__(self, *size, bins, k=2) -> None:
        self.num_bins = bins
        self.fn = KWiseHash(*size, k=k)

    def __call__(self, items: torch.Tensor) -> torch.Tensor:
        return self.fn.bin(items, self.num_bins)

MethodName = Literal["exact", "ams", "compass-merge", "compass-partition", "count-conv","our-method"]

class Arguments(Tap):
    method: MethodName # Use count-conv for our proposed method
    query: str # For the list of available queries see the README
    seed: Optional[int] = None
    bins: int = 1
    means: int = 1
    medians: int = 1
    estimates: int = 1
    batch_size: int = 2**16
    result_dir: str = "results"
    data_dir: str = ""
    topk: int = 0

    def process_args(self):
        # Validate arguments
        if self.method == "ams" and self.bins != 1:
            raise ValueError("Bins must be 1 for AMS")

        if self.method == "ams":
            self.batch_size = max(self.batch_size // self.means, 1)

        if self.method == "ams" and self.bins != 1:
            raise ValueError("bins must be 1 for the ams method")

        if self.method != "ams" and self.means != 1:
            raise ValueError("means can only be used with the ams methods")
        
        if self.method == "exact":
            if self.bins != 1 or self.means != 1 or self.medians != 1 or self.estimates != 1:
                raise ValueError("bins, means, medians, and estimates must be 1 for the exact method")

        if self.bins < 1:
            raise ValueError("Number of bins cannot be negative")

        if self.means < 1:
            raise ValueError("Number of means cannot be negative")

        if self.medians < 1:
            raise ValueError("Number of medians cannot be negative")

        if self.estimates < 1:
            raise ValueError("Number of estimates cannot be negative")

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

def read_sql_query(root: str, benchmark: str, name: str) -> str:

    if benchmark == "job_light":
        path = os.path.join(
            root, CARDEST_DIR, "workloads", "job-light", "job_light_queries.sql"
        )
    elif benchmark == "job_light_sub":
        path = os.path.join(
            root,
            CARDEST_DIR,
            "workloads",
            "job-light",
            "sub_plan_queries",
            "job_light_sub_query.sql",
        )
    elif benchmark == "stats":
        path = os.path.join(
            root, CARDEST_DIR, "workloads", "stats_CEB", "stats_CEB.sql"
        )
    elif benchmark == "stats_sub":
        path = os.path.join(
            root,
            CARDEST_DIR,
            "workloads",
            "stats_CEB",
            "sub_plan_queries",
            "stats_CEB_sub_queries.sql",
        )
    else:
        raise ValueError(f"Query '{benchmark}-{name}' does not exist.")

    with open(path, "r") as f:
        if benchmark == "job":
            sql = f.read()

        else:
            sqls = f.readlines()
            idx = int(name) - 1

            if idx < 0 or idx >= len(sqls):
                raise ValueError(f"Query '{benchmark}-{name}' does not exist.")

            sql = sqls[idx]

            # stats has the true cardinality prepended to the query
            if benchmark == "stats":
                sql = sql.split("||", 1)[1]
            # stats_sub has the corresponding full query appended to the query
            elif benchmark == "stats_sub":
                sql = sql.split("||", 1)[0]

    return sql.strip()

class Tokenizer(object):
    token2idx: dict[str, int]
    idx2token: list[str]

    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add(self, token: str) -> int:
        """Adds a token to the dictionary and returns its index

        For any input that is not a string, returns NaN.

        Args:
            token (str): the token to add

        Returns:
            int: the index of the token
        """
        if type(token) != str:
            return float("nan")

        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1

        return self.token2idx[token]

    def __getitem__(self, index_or_token: Union[int, str]) -> Union[str, int]:
        """Returns the token at the given index or the index of the given token"""
        if type(index_or_token) == int:
            return self.idx2token[index_or_token]
        else:
            return self.token2idx[index_or_token]

    def __len__(self) -> int:
        return len(self.idx2token)

class Table(object):
    name: str
    attributes: List[str]
    attribute2idx: Dict[str, int]
    data: torch.Tensor
    tokenizers: Dict[str, Tokenizer]

    def __init__(
        self,
        df: pd.DataFrame,
        name: str,
        attributes: List[str],
        string_attributes: List[str],
        datetime_attributes: List[str],
    ) -> None:

        self.name = name
        self.attributes = attributes
        self.attribute2idx = {name: i for i, name in enumerate(self.attributes)}
        self.tokenizers = {}

        for attr in datetime_attributes:
            self._datetime_attr(df, attr)

        for attr in string_attributes:
            self._tokenize_attr(df, attr)

        self.data = torch.as_tensor(df.values, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_records

    def __repr__(self) -> str:
        attributes = ", ".join(self.attributes)
        return f"{self.name}({attributes})"

    @property
    def num_records(self) -> int:
        return self.data.size(0)

    @property
    def num_attributes(self) -> int:
        return self.data.size(1)

    @staticmethod
    def datetime2int(date: str, format: Optional[str] = None) -> int:
        return int(pd.to_datetime(date, format=format).timestamp())

    def _datetime_attr(self, df: pd.DataFrame, attribute: str) -> None:
        df[attribute] = df[attribute].astype("int64") // 10**9

    def _tokenize_attr(self, df: pd.DataFrame, attribute: str) -> None:
        # create dictionary mapping unique values to integers
        # map over rows and replace values with integers
        dictionary = Tokenizer()
        df[attribute] = df[attribute].apply(lambda x: dictionary.add(x))
        self.tokenizers[attribute] = dictionary

class Query(object):
    sql: str
    joins: List[Tuple[str, str, str]]
    selects: List[Tuple[str, str, str]]
    node2component: Dict[str, int]
    num_components: int
    id2joined_attrs: Dict[str, Set[str]]

    def __init__(self, sql: str):
        self.sql = sql

        self.joins = []
        self.selects = []

        for left, op, right, is_select in self.condition_iter():
            if is_select:
                self.selects.append((left, op, right))
            else:
                self.joins.append((left, op, right))

        self.node2component, self.num_components = self.component_labeling(self.joins)

        self.id2joined_attrs: Dict[str, Set[str]] = defaultdict(lambda: set())

        for join in self.joins:
            left, _, right = join

            id, attr = left.split(".")
            self.id2joined_attrs[id].add(attr)

            id, attr = right.split(".")
            self.id2joined_attrs[id].add(attr)

    def __repr__(self) -> str:
        return self.sql

    def table_mapping_iter(self) -> Generator[Tuple[str, str], None, None]:

        table_list = text_between(self.sql, "FROM", "WHERE")
        table_list = table_list.split(",")

        for table in table_list:
            table = table.strip()
            
            # First try splitting on AS otherwise split on space
            splits = re.split(" AS ", table, flags=re.IGNORECASE, maxsplit=1)
            if len(splits) == 1:
                splits = table.split(" ", maxsplit=1)
            
            name, id = splits

            name = name.strip()
            id = id.strip()

            yield id, name

    def condition_iter(self) -> Generator[Tuple[str, str, str, bool], None, None]:

        # remove closing semicolon if present
        if self.sql.endswith(";"):
            sql_query = self.sql[:-1]
        else:
            sql_query = self.sql

        selections = re.split("\sWHERE\s", sql_query)[1]

        if " OR " in selections:
            raise NotImplementedError("OR selections are not supported yet.")

        selections = re.split("\sAND\s", selections)
        
        # TODO support more complicated LIKE and OR statements
        # TODO support for parentheses

        for i, selection in enumerate(selections):
            left, op, right = selection_ops_re.split(selection)
            left = left.strip()
            right = right.strip()

            # With BETWEEN the next AND is part of BETWEEN
            if op == "BETWEEN":
                right += " AND " + selections[i + 1].strip()
                selections.pop(i + 1)

            is_selection = attribute_re.match(right) == None

            if attribute_re.match(left) == None:
                raise NotImplementedError(
                    "Selection values on the left are not supported"
                )

            if not is_selection and op != "=":
                raise ValueError(f"Must be equi-join but got: {op}")

            yield left, op, right, is_selection

    def component_labeling(self, joins: List[Tuple[str, str, str]]) -> Dict[str, int]:
        to_visit: Set[str] = set()
        node2component: Dict[str, int] = {}
        num_components = 0

        for join in joins:
            left, op, right = join

            to_visit.add(left)
            to_visit.add(right)

        def depth_first_search(node: str, component: int):
            node2component[node] = component

            for join in joins:
                left, op, right = join

                # get the other node if this join involves the current node
                # if not then continue to the next join
                if left == node:
                    other = right
                elif right == node:
                    other = left
                else:
                    continue

                # if the other node has already been visited then continue
                if other not in to_visit:
                    continue

                to_visit.remove(other)
                depth_first_search(other, component)

        while len(to_visit) > 0:
            node = to_visit.pop()
            depth_first_search(node, num_components)
            num_components += 1

        return node2component, num_components

    def joins_of(self, table_id: str) -> List[Tuple[str, str, str]]:
        # ensures that left always has the table id attribute
        joins = []

        for join in self.joins:
            left, op, right = join

            id, _ = left.split(".")
            if id == table_id:
                joins.append(join)

            id, _ = right.split(".")
            if id == table_id:
                joins.append((right, op, left))

        return joins

    def joined_nodes(self, table_id: str) -> Set[str]:
        nodes: Set[str] = set()

        for join in self.joins:
            left, _, right = join

            id, _ = left.split(".")
            if id == table_id:
                nodes.add(left)

            id, _ = right.split(".")
            if id == table_id:
                nodes.add(right)

        return nodes

    def joined_with(self, node: str) -> Set[str]:
        nodes: Set[str] = set()

        for join in self.joins:
            left, _, right = join

            if left == node:
                nodes.add(right)

            if right == node:
                nodes.add(left)

        return nodes

    def joined_with_and_idx(self, node: str) -> Dict[str, int]:
        nodes: Dict[str, int] = {}
        
        for join_idx,join in enumerate(self.joins):
            left, _, right = join
            
            if left == node:
                nodes[right] = join_idx
                
            if right == node:
                nodes[left] = join_idx 
                
        return nodes
    
    def random_node(self) -> str:
        nodes = list(self.node2component.keys())
        idx = random.randint(0, len(nodes) - 1)
        return nodes[idx]

def load_tables(root: str, benchmark: str, query: Query) -> Dict[str, Table]:

    id2table: Dict[str, Table] = {}

    # Read the SQL definitions of the tables

    if benchmark.startswith("job"):
        schema_path = os.path.join(root, IMDB_DIR, "schematext.sql")
    elif benchmark.startswith("stats"):
        schema_path = os.path.join(
            root, CARDEST_DIR, "datasets", "stats_simplified", "stats.sql"
        )
    else:
        raise ValueError(f"Benchmark '{benchmark}' does not exist.")

    with open(schema_path, "r") as f:
        sql = f.read()

    # Load only each table in the SQL query once

    for id, name in query.table_mapping_iter():

        # For each table check if it was already loaded
        # because we can reference the same data table multiple times

        table = None
        for t in id2table.values():
            if t.name == name:
                table = t
                break

        if table != None:
            id2table[id] = table
            continue

        # If the table was not loaded already load it,
        # try loading it from a pickle cache (faster)

        if benchmark.startswith("job"):
            pickle_path = os.path.join(CACHE_DIR, "imdb", name + ".pkl")
        elif benchmark.startswith("stats"):
            pickle_path = os.path.join(CACHE_DIR, "stats", name + ".pkl")
        else:
            raise ValueError(f"Benchmark '{benchmark}' does not exist.")
        
        if os.path.isfile(pickle_path):
            print("Using cached table:", pickle_path)
            table = torch.load(pickle_path)
            id2table[id] = table
            continue

        # Otherwise, load it from the csv files (slower)

        # Read the SQL definition of the table
        idx = sql.index(f"CREATE TABLE {name}")
        idx_start = sql.index("(", idx)
        idx_end = sql.index(");", idx)
        attributes = sql[idx_start + 1 : idx_end]
        attributes = attributes.split(",")
        # Creates list of (attribute_name, type)
        attributes = [tuple(a.strip().split(" ", 1)) for a in attributes]

        if benchmark.startswith("job"):
            data_path = os.path.join(root, IMDB_DIR, name + ".csv")
        elif benchmark.startswith("stats"):
            data_path = os.path.join(
                root, CARDEST_DIR, "datasets", "stats_simplified", name + ".csv"
            )
        else:
            raise ValueError(f"Benchmark '{benchmark}' does not exist.")

        attribute_names = [a[0] for a in attributes]
        string_attributes = [
            a[0] for a in attributes if a[1].upper().startswith("CHARACTER")
        ]
        datetime_attributes = [a[0] for a in attributes if a[1] == "TIMESTAMP"]

        dtype_mapping = {
            "CHARACTER": str,
            "TIMESTAMP": str,
            "INTEGER": float,
            "SERIAL": float,
            "SMALLINT": float,
        }

        dtypes = {a[0]: dtype_mapping[a[1].upper().split(" ")[0]] for a in attributes}

        if benchmark.startswith("stats"):
            df = pd.read_csv(
                data_path,
                header=0,
                parse_dates=datetime_attributes,
                encoding='utf-8', 
                sep=",",
                names=attribute_names,
                dtype=dtypes,
            )

        elif benchmark.startswith("job"):
            with open(data_path, "r") as f:
                # TODO: memory usage could be improved by replacing unescaped variable
                # in an iterator fashion instead of all at ones.
                data = f.read()
                
            # Replace the escaped quotes by double quotes to fix parsing errors
            data = escaped_backslash_re.sub("\"\"", data)

            # These lines cause trouble because they end with a backslash before the final quote
            if name == "movie_info":
                data = data.replace(
                    "'Harry Bellaver' (qv).\\\"\",",
                    "'Harry Bellaver' (qv).\\\\\","
                )
                data = data.replace(
                    "who must go back and find his bloodlust one last time. \\\"\",",
                    "who must go back and find his bloodlust one last time. \\\\\","
                )

            elif name == "person_info":
                data = data.replace("\\\"\",", "\\\\\",")

            df = pd.read_csv(
                io.StringIO(data),
                header=None,
                parse_dates=datetime_attributes,
                encoding='utf-8', 
                sep=",",
                names=attribute_names,
                dtype=dtypes,
            )

        null_occurences = df.isin([NULL_VALUE]).values.sum()
        if null_occurences > 0:
            raise RuntimeError(
                f"Found the NULL value in the table {name}, consider using a different NULL value."
            )

        # arbitrary value used to denote NULL
        df.fillna(NULL_VALUE, inplace=True)

        table = Table(df, name, attribute_names, string_attributes, datetime_attributes)
        id2table[id] = table

    return id2table

def make_selection_filters(
    id2table: Dict[str, Table], query: Query
) -> Dict[str, torch.Tensor]:
    id2mask: Dict[str, torch.Tensor] = {}

    for select in query.selects:
        left, op, right = select
        id, attr = left.split(".")
        table = id2table[id]
        attr_idx = table.attribute2idx[attr]

        if right.endswith("::timestamp"):
            timestamp = right[1 : -len("'::timestamp")]
            value = table.datetime2int(timestamp)

        elif is_number(right):
            value = float(right) if "." in right else int(right)

        elif right.startswith(("'", '"')) and right.endswith(("'", '"')):
            value = table.tokenizers[attr][right[1:-1]]

        else:
            raise ValueError(f"Not sure how to handle right value: {right}")

        if op == "=":
            mask = table.data[:, attr_idx] == value
        elif op == "<>":
            mask = table.data[:, attr_idx] != value
        elif op == ">":
            mask = table.data[:, attr_idx] > value
        elif op == "<":
            mask = table.data[:, attr_idx] < value
        elif op == "<=":
            mask = table.data[:, attr_idx] <= value
        elif op == ">=":
            mask = table.data[:, attr_idx] >= value

        # Ensure that the NULL values are removed from the column
        # because any condition with NULL is false
        mask &= table.data[:, attr_idx] != NULL_VALUE

        if id in id2mask:
            # Assumes all the selections are AND together
            id2mask[id] &= mask
        else:
            id2mask[id] = mask

    # Ensure that the NULL values are removed from the joined columns
    for join in query.joins:
        left, op, right = join

        id, attr = left.split(".")
        table = id2table[id]
        attr_idx = table.attribute2idx[attr]

        mask = table.data[:, attr_idx] != NULL_VALUE

        if id in id2mask:
            # Assumes all the selections are AND together
            id2mask[id] &= mask
        else:
            id2mask[id] = mask

        id, attr = right.split(".")
        table = id2table[id]
        attr_idx = table.attribute2idx[attr]

        mask = table.data[:, attr_idx] != NULL_VALUE

        if id in id2mask:
            # Assumes all the selections are AND together
            id2mask[id] &= mask
        else:
            id2mask[id] = mask

    return id2mask

def prepare_batches(
    id2table: Dict[str, Table],
    id2mask: Dict[str, torch.Tensor],
    batch_size: int,
    query: Query,
) -> Dict[str, List[torch.Tensor]]:

    node2batches: Dict[str, List[torch.Tensor]] = {}

    # Capture the set of all unique nodes
    nodes: Set[str] = set()
    for join in query.joins:
        left, _, right = join
        nodes.add(left)
        nodes.add(right)

    # For each node load its data
    for node in nodes:

        id, attr = node.split(".")
        table = id2table[id]

        attr_idx = table.attribute2idx[attr]
        attr_data = table.data[:, attr_idx]

        mask = id2mask.get(id, None)
        if mask != None:
            attr_data = attr_data[mask]

        attr_batches = attr_data.split(batch_size)
        node2batches[node] = attr_batches
    return node2batches
    
def combine_sketches(
    node: str, visited: Set[str], query: Query, id2sketch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    id, _ = node.split(".")
    sketch = id2sketch[id]
    visited.add(node)

    for other_node in query.joined_nodes(id):
        # skip the current node
        if other_node == node:
            continue

        visited.add(other_node)

        tmp = 1
        for joined_node in query.joined_with(other_node):
            tmp = tmp * combine_sketches(joined_node, visited, query, id2sketch)

        # efficient circular cross-correlation
        sketch = ifft(fft(tmp).conj() * fft(sketch)).real

    for joined_node in query.joined_with(node).difference(visited):
        sketch = sketch * combine_sketches(joined_node, visited, query, id2sketch)

    return sketch

def median_trick(estimates: torch.Tensor, medians: int) -> torch.Tensor:
    """Takes a tensor of iid estimates and returns the median among groups"""
    if medians == 1:
        return estimates

    estimates = estimates.view(-1, medians)
    return torch.median(estimates, dim=1).values
