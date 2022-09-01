from .fits import read_map_result_from_fits, write_map_result_to_fits

IO_FORMATS_MAP_RESULT_READ = {"fits": read_map_result_from_fits}
IO_FORMATS_MAP_RESULT_WRITE = {"fits": write_map_result_to_fits}