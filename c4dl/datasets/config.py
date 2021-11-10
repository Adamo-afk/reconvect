from .composite import CompositeReader
from .. import projection


# new reader classes can be added here at runtime without modifying this file
reader_classes = {}


def reader_class(data_source):
    # use this structure instead of a dict so we can do lazy imports
    if data_source == "aster":
        from .asterdem import ASTERDEMReader
        return ASTERDEMReader
    elif data_source == "ecmwf":
        from .ecmwfnwp import ECMWFNWPReader
        return ECMWFNWPReader
    elif data_source == "cosmoccs4":
        from .cosmoccs4 import COSMOCCS4Reader
        return COSMOCCS4Reader
    elif data_source == "goesabi":
        from .goesabi import GOESABIReader
        return GOESABIReader
    elif data_source == "goesglm":
        from .goesglm import GOESGLMReader
        return GOESGLMReader
    elif data_source == "mchlightning":
        from .mchlightning import MCHLightningReader
        return MCHLightningReader
    elif data_source == "mchradar":
        from .mchradar import MCHRadarReader
        return MCHRadarReader
    elif data_source == "msg":
        from .msg import MSGRadianceCCS4Reader
        return MSGRadianceCCS4Reader
    elif data_source == "nexrad":
        from .nexrad import NEXRADReader
        return NEXRADReader
    elif data_source == "swissdem":
        from .swissdem import SwissDEMReader
        return SwissDEMReader
    else:
        return reader_classes[data_source]


def prepare_readers(config):
    area = config["area"]
    grid_projection = projection.GridProjection(area)
    data_sources_config = config["data_sources"]
    reader_instances = {
        data_source: 
        prepare_reader(data_source, data_sources_config[data_source],
            grid_projection)
        for data_source
        in data_sources_config
    }

    if config.get("composites"):
        reader_instances["composites"] = prepare_composites(
            reader_instances, config)

    return (grid_projection, reader_instances)


def prepare_reader(data_source, config, grid_projection):
    reader_cls = reader_class(data_source)
    return reader_cls(grid_projection, **config)


def prepare_composites(readers, config):
    from . import composite
    composite_cfg = config["composites"]
    composite_spec = {}
    for (name,cfg) in composite_cfg.items():
        composite_spec[name] = {}
        composite_spec[name]["type"] = cfg["type"]
        composite_spec[name]["reader_vars"] = prepare_composite_vars(
            readers, cfg["params"])
    return CompositeReader(composite_spec)
    
    
def prepare_composite_vars(readers, params):
    params = params.split(" ")
    reader_vars = (p.split("::") for p in params)
    return [(readers[r], v) for (r,v) in reader_vars]
