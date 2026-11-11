from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.modules.ee_modules import (
        EXITS_PLACEMENT_PRESETS,
        POSSIBLE_EXITS,
        EEModelOutput,
        EESequenceClassifierOutput,
        EETrainer,
        EETrainingArguments,
        EarlyExitException,
        EarlyExitHead,
        EarlyExitInference,
        EarlyExitPolicy,
        EarlyExitStrategy,
        ExitConfig,
        LogitNormLoss,
        TemperatureScaler,
        compute_ece,
        compute_entropy_reg_weights,
        entropy,
        max_confidence,
        params_per_exit,
        top12_margin,
    )
    from model.modules.layoutlmv3 import (
        LayoutLMv3Backbone,
        LayoutLMv3BackboneOutput,
        LayoutLMv3EEForSequenceClassification,
        LayoutLMv3EncoderEE,
        LayoutLMv3Exit,
        LayoutLMv3ModelEE,
    )
    from model.modules.ocr_ppdoclayout_ppocrv6small import DocumentOCR
    from model.modules.ocr_pytesseract import TesseractOCR
    from model.modules.projector import SupConProjector

  

  
_EE_EXPORTS = {
    "POSSIBLE_EXITS",
    "EXITS_PLACEMENT_PRESETS",
    "EarlyExitStrategy",
    "EarlyExitInference",
    "EarlyExitHead",
    "ExitConfig",
    "EarlyExitException",
    "EEModelOutput",
    "EESequenceClassifierOutput",
    "LogitNormLoss",
    "EETrainingArguments",
    "EETrainer",
    "TemperatureScaler",
    "EarlyExitPolicy",
    "compute_ece",
    "compute_entropy_reg_weights",
    "params_per_exit",
    "entropy",
    "max_confidence",
    "top12_margin",
}

_LAYOUTLMV3_EXPORTS = {
    "LayoutLMv3Exit",
    "LayoutLMv3EncoderEE",
    "LayoutLMv3ModelEE",
    "LayoutLMv3EEForSequenceClassification",
    "LayoutLMv3Backbone",
    "LayoutLMv3BackboneOutput",
}


def __getattr__(name: str):
    if name in _EE_EXPORTS:
        from model.modules import ee_modules
        return getattr(ee_modules, name)
    if name in _LAYOUTLMV3_EXPORTS:
        from model.modules import layoutlmv3
        return getattr(layoutlmv3, name)
    if name == "DocumentOCR":
        from model.modules.ocr_ppdoclayout_ppocrv6small import DocumentOCR
        return DocumentOCR
    if name == "TesseractOCR":
        from model.modules.ocr_pytesseract import TesseractOCR
        return TesseractOCR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name=="SupConProjector":
        from model.modules.projector import SupConProjector
        return SupConProjector

__all__ = [
    *_EE_EXPORTS,
    *_LAYOUTLMV3_EXPORTS,
    "DocumentOCR",
    "TesseractOCR",
]
