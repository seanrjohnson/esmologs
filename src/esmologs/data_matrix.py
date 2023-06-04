"""

The DataMatrix class is an abstraction around dense and sparse matrices

"""

from os import PathLike
import pandas as pd
import scipy.sparse
import numpy as np
import h5py
from typing import Dict, List, Tuple, Union
import warnings
from pathlib import Path

EXTENSION_TO_TYPE = {
    "gb":"genbank",
    "gbk":"genbank",
    "genbank":"genbank",
    "fasta":"fasta",
    "fa":"fasta",
    "fna":"fasta",
    "pep":"fasta",
    # "gff":"gff",
    # "gtf":"gff",
    # "gff3":"gff",
    "hmmer":"hmm",
    "hmm":"hmm",
    "hdf5":"hdf5",
    "hdf":"hdf5",
    "h5":"hdf5",
}
def get_file_type(filename):
    out = None
    file_extension = Path(filename).suffix[1:]
    if file_extension in EXTENSION_TO_TYPE:
        out = EXTENSION_TO_TYPE[file_extension]
    return out

#TODO: switch to scipy sparse array instead of sparse matrix
#TODO: use hdf5plugin to blosc compress datasets: http://www.silx.org/doc/hdf5plugin/latest/usage.html

UTF8_h5py_encoding = h5py.string_dtype(encoding='utf-8')

#TODO: maybe sub-classes for dense vs sparse?
#TODO: maybe add an attr for value label

class DataMatrix():
    """
    Class used to abstract access to sparse and dense matrices

    Dense matrices are stored as numpy arrays
    Sparse matrices are stored as scipy.sparse.csr_matrix
    Intended to be read-only
    
    """
    _ROW_LABELS_DATASET="ROW_LABELS"
    """string array"""
    _COL_LABELS_DATASET="COL_LABELS" 
    """string array (only used if SYMMETRIC LABELS is not True)"""
    _DENSE_DATASET="DENSE_DATA"
    """2d array of floats, for dense data"""
    _SPARSE_VALUES_DATASET="SPARSE_VALUES" 
    """values for CSR sparse"""
    _SPARSE_CSR_INDICES_DATASET="SPARSE_CSR_INDICES"
    """row index for CSR sparse"""
    _SPARSE_CSR_INDPTR_DATASET="SPARSE_CSR_INDPTR"
    """col index for CSR sparse"""
    _ARRAY_TYPE_ATTR="ARRAY_TYPE"
    """{DENSE, SPARSE_CSR}, sparse implies CSR"""
    _SYMMETRIC_LABELS_ATTR="SYMMETRIC_LABELS"
    """if the x-axis and y-axis have the same labels"""
    _DOMAINATOR_MATRIX_FILE_VERSION_ATTR="DOMAINATOR_MATRIX_FILE_VERSION"


    _DENSE = "DENSE"
    _SPARSE_CSR = "SPARSE_CSR"
    _POSSIBLE_ARRAY_TYPES={_DENSE, _SPARSE_CSR}
    
    _DOMAINATOR_MATRIX_FILE_VERSION = "1.0"  #TODO: increment after any breaking changes

    def __init__(self, data=None, row_names:Dict[str, int]=None, col_names:Dict[str, int]=None):
        """
            

        """

        # define instance attributes
        self.sparse = False
        self.rows = None
        self.columns = None
        self.row_to_idx = None
        self.column_to_idx = None
        self.data = None

        if data is not None:
            rows_list, cols_list = self.validate_matrix_data(data, row_names, col_names)
            self.rows = rows_list
            self.columns = cols_list
            self.row_to_idx = row_names.copy() #TODO: should this be a copy?
            if row_names == col_names:
                self.column_to_idx = self.row_to_idx
            else:
                self.column_to_idx = col_names.copy() #TODO: should this be a copy?
            
            if isinstance(data, scipy.sparse.spmatrix):
                self.sparse = True
                if type(type) != scipy.sparse.csr_matrix:
                    self.data = scipy.sparse.csr_matrix(data)
                else:
                    self.data = data #TODO: should this be a copy?
            else: #dense
                self.data = data #TODO: should this be a copy?

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def symmetric(self):
        return self.rows == self.columns

    def __len__(self):
        return self.data.shape[0]

    @classmethod
    def from_file(cls, matrix_file):
        out = cls()
        file_type = get_file_type(matrix_file)
        if file_type == "hdf5": 
            out._read_hdf5(matrix_file)
        else: #assume dense tsv text file as parse as such
            out._read_text_dense_matrix(matrix_file) 
        return out

    def _read_hdf5(self, matrix_file: Union[PathLike, str]):
        
        f = h5py.File(matrix_file)
        
        ### validate file format  ###
        for attribute in (self._DOMAINATOR_MATRIX_FILE_VERSION_ATTR, self._ARRAY_TYPE_ATTR, self._SYMMETRIC_LABELS_ATTR):
            if attribute not in f.attrs:
                raise ValueError(f"attribute {attribute} not found in hdf5 file. Are you sure it is a Domainator formatted file?")
        file_version = f.attrs[self._DOMAINATOR_MATRIX_FILE_VERSION_ATTR]
        if file_version != self._DOMAINATOR_MATRIX_FILE_VERSION:
            warnings.warn(f"Matrix file version ({file_version}) in file {matrix_file} does not match program-supported version ({self._DOMAINATOR_MATRIX_FILE_VERSION})")
        array_type = f.attrs[self._ARRAY_TYPE_ATTR]
        if array_type not in self._POSSIBLE_ARRAY_TYPES:
            raise ValueError(f"Array type {array_type} not recognized")

        if array_type == "DENSE":
            for dataset_name in (self._ROW_LABELS_DATASET, self._DENSE_DATASET):
                if dataset_name not in f:
                    raise ValueError(f"expected dataset {self._DENSE_DATASET} not found in file {matrix_file}")
        elif array_type == "SPARSE_CSR": # sparse
            for dataset_name in (self._ROW_LABELS_DATASET, self._SPARSE_VALUES_DATASET, self._SPARSE_CSR_INDICES_DATASET, self._SPARSE_CSR_INDPTR_DATASET):
                if dataset_name not in f:
                    raise ValueError(f"expected dataset {dataset_name} not found in file {matrix_file}")

        if not f.attrs[self._SYMMETRIC_LABELS_ATTR]:
            if self._COL_LABELS_DATASET not in f:
                raise ValueError(f"expected dataset {self._COL_LABELS_DATASET} not found in file {matrix_file}")
        

        ### store data ###
        self.sparse = False
        self.rows = None
        self.columns = None
        self.row_to_idx = None
        self.column_to_idx = None
        self.data = None

        self.rows, self.row_to_idx = self.parse_axis_index(f[self._ROW_LABELS_DATASET])

        if f.attrs[self._SYMMETRIC_LABELS_ATTR]:
            self.columns = self.rows
            self.column_to_idx = self.row_to_idx
        else:
            self.columns, self.column_to_idx = self.parse_axis_index(f[self._COL_LABELS_DATASET])

        if f.attrs[self._ARRAY_TYPE_ATTR] == "DENSE":
            self.sparse = False
            self.data = f[self._DENSE_DATASET][:]
        elif f.attrs[self._ARRAY_TYPE_ATTR] == "SPARSE_CSR":
            self.sparse = True
            self.data = scipy.sparse.csr_matrix((f[self._SPARSE_VALUES_DATASET][:], f[self._SPARSE_CSR_INDICES_DATASET][:], f[self._SPARSE_CSR_INDPTR_DATASET][:]), (len(self.rows), len(self.columns)) )

    
    def iter_data(self): #TODO: figure out why sparse matrices iterate in a different order than dense matrices. For example, FeSOD_dist.dense.hdf5 vs FeSOD_dist.sparse.hdf5
        """
            yields (row_name, column_name, value).
            for dense matrices iterates over every value in the matrix
            for sparse matrices, skips data that is not represented in the data structure.
        """

        if not self.sparse:
            for r in range(self.data.shape[0]):
                for c in range(self.data.shape[1]):
                    yield self.rows[r], self.columns[c], self.data[r,c]

        else: 
            #https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
            for r in range(self.data.shape[0]):
                for ind in range(self.data.indptr[r], self.data.indptr[r+1]):
                    yield self.rows[r], self.columns[self.data.indices[ind]], self.data.data[ind]

    def itervalues(self):
        """ yield values from the matrix one by one by row then column.
        
        """

        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                yield self.data[r,c]
    
    def save(self, file):
        """write the matrix to an hdf5 file

        Args:
            file (path or string): 
        """

        if self.sparse:
            self.write_sparse(self.data, file, self.row_to_idx, self.column_to_idx)
        else: # dense
            self.write_dense(self.data, file, self.row_to_idx, self.column_to_idx)


    def convert_to_sparse(self):
        """Inplace conversion to csr sparse

        """
        if self.sparse:
            pass
        else:
            self.sparse = True
            self.data = scipy.sparse.csr_matrix(self.data)

    def convert_to_dense(self):
        """Inplace conversion to dense

        """
        if self.sparse:
            self.sparse = False
            self.data = self.data.toarray()
        else:
            pass

    def toarray(self):
        if self.sparse:
            return self.data.toarray()
        else:
            return self.data

    def _read_text_dense_matrix(self, matrix_file: Union[PathLike, str]):
        self.sparse = False
        try:
            mat_file = pd.read_csv(matrix_file, sep="\t", index_col=0)
        except UnicodeDecodeError:
            raise ValueError(f"Expecting utf-8 encoded matrix file, but found invalid unicode byte. You may be trying to pass in an hdf5 file, but using an unrecognized extension.")
        self.rows = list(mat_file.index)
        self.columns = list(mat_file.columns)

        self.row_to_idx = {v:i for i,v in enumerate(self.rows)}
        self.column_to_idx = {v:i for i,v in enumerate(self.columns)}

        if all(a == b for a,b in zip(self.rows, self.columns)):
            self.columns = self.rows

        self.data = mat_file.to_numpy()

    @staticmethod
    def parse_axis_index(axis_index):
        items_list = [x for x in axis_index]
        items_dict = {v:i for i,v in enumerate(items_list)}
        
        return items_list, items_dict



    @classmethod
    def write_sparse(cls, matrix, file_name, row_names, col_names):
        """Writes a dok sparse matrix to an hdf5 file
        
        """

        # check validity of input data
        row_names, col_names = cls.validate_matrix_data(matrix, row_names, col_names)
        
        if type(matrix) != scipy.sparse.csr_matrix:
            matrix = scipy.sparse.csr_matrix(matrix)

        f = h5py.File(file_name, 'w')
        f.attrs[cls._DOMAINATOR_MATRIX_FILE_VERSION_ATTR] = cls._DOMAINATOR_MATRIX_FILE_VERSION
        f.attrs[cls._ARRAY_TYPE_ATTR] = cls._SPARSE_CSR

        f.create_dataset(cls._SPARSE_VALUES_DATASET, data=matrix.data)
        f.create_dataset(cls._SPARSE_CSR_INDICES_DATASET, data=matrix.indices)
        f.create_dataset(cls._SPARSE_CSR_INDPTR_DATASET, data=matrix.indptr)

        f.create_dataset(cls._ROW_LABELS_DATASET, data=row_names, dtype=UTF8_h5py_encoding)
        if row_names is col_names:
            f.attrs[cls._SYMMETRIC_LABELS_ATTR] = True
        else:
            f.attrs[cls._SYMMETRIC_LABELS_ATTR] = False
            f.create_dataset(cls._COL_LABELS_DATASET, data=col_names, dtype=UTF8_h5py_encoding)
        
        f.close()


    @staticmethod
    def names_dict_to_list(names_dict:Dict[str, int]) -> List[str]:
        """Converts a dict of indexes into a list of keys

        Args:
            names_dict (Dict[str, int]): {row_name: row_index}

        Returns:
            List[str]: List of strings where list[row_index] = row_name
        """
        out = [None] * len(names_dict)
        for name, idx in names_dict.items():
            out[idx] = name
        if any(v is None for v in out):
            raise ValueError("names_dict is not a unique mapping.")
        return out

    @classmethod
    def validate_matrix_data(cls, matrix: Union[np.array, scipy.sparse.spmatrix], row_names:List[str], col_names:List[str]) -> Tuple[List[str], List[str]]:
        """Check for consistency between a matrix and row/column labels

        Args:
            matrix (Union[np.array, scipy.sparse.spmatrix]): dense or sparse (dok) data matrix
            row_names:List[str]: row labels
            col_names:List[str]: column labels

        """

        if len(matrix.shape) != 2:
            raise ValueError(f"matrix must be 2-dimensional, not: {matrix.shape}")
        
        if len(row_names) != matrix.shape[0]:
            raise ValueError(f"row_names size must match dense_matrix dim 0, not: {len(row_names)} vs. {matrix.shape[0]}")

        if len(col_names) != matrix.shape[1]:
            raise ValueError(f"col_names size must match dense_matrix dim 1, not: {len(col_names)} vs. {matrix.shape[1]}")


        if (row_names == col_names) or (all(a == b for a,b in zip(row_names, col_names))):
            col_names = row_names

        return row_names, col_names        


    @classmethod
    def write_dense(cls, matrix:Union[np.array, scipy.sparse.spmatrix], file_name: Union[PathLike, str], row_names:List[str], col_names:List[str]):
        """Writes a dense matrix to an hdf5 file.
        
        Args:
            matrix (Union[np.array, scipy.sparse.spmatrix]): 2d array
            file_name (Union[PathLike, str]): path to the target file to write the data to
            row_names:List[str]: row labels
            col_names:List[str]: column labels
        """
        # check validity of input data
        if isinstance(matrix, scipy.sparse.spmatrix):
            matrix = matrix.toarray()

        row_names, col_names = cls.validate_matrix_data(matrix, row_names, col_names)
        

        f = h5py.File(file_name, 'w')
        f.attrs[cls._DOMAINATOR_MATRIX_FILE_VERSION_ATTR] = cls._DOMAINATOR_MATRIX_FILE_VERSION
        f.attrs[cls._ARRAY_TYPE_ATTR] = cls._DENSE
        f.create_dataset(cls._DENSE_DATASET, data=matrix)
        
        
        f.create_dataset(cls._ROW_LABELS_DATASET, data=row_names, dtype=UTF8_h5py_encoding)
        if row_names is col_names:
            f.attrs[cls._SYMMETRIC_LABELS_ATTR] = True
        else:
            f.attrs[cls._SYMMETRIC_LABELS_ATTR] = False
            f.create_dataset(cls._COL_LABELS_DATASET, data=col_names, dtype=UTF8_h5py_encoding)
            
        f.close()

    @classmethod
    def write_dense_text(cls, matrix:Union[np.array, scipy.sparse.spmatrix], file_name: Union[PathLike, str], row_names:List[str], col_names:List[str]):
        """Writes a dense matrix to an hdf5 file.
        
        Args:
            matrix (Union[np.array, scipy.sparse.spmatrix]): 2d array
            file_name (Union[PathLike, str]): path to the target file to write the data to
            row_names:List[str]: row labels
            col_names:List[str]: column labels
        """

        row_names, col_names = cls.validate_matrix_data(matrix, row_names, col_names)
        outfile = open(file_name, "w")
        print("\t" + "\t".join(col_names), file=outfile)
        
        if isinstance(matrix, scipy.sparse.spmatrix):
            for idx in range(len(row_names)):
                print(row_names[idx] + "\t" + "\t".join([str(x) for x in np.nditer(matrix[idx].todense())]), file=outfile)
        else:
            for idx in range(len(row_names)):
                print(row_names[idx] + "\t" + "\t".join([str(x) for x in np.nditer(matrix[idx])]), file=outfile)

        outfile.close()
