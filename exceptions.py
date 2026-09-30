class IngestionError(Exception):
    """Base ingestion exception."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when the file type is not supported."""


class FileValidationError(IngestionError):
    """Raised when a file fails validation."""


class DocumentParsingError(IngestionError):
    """Raised when parsing fails."""