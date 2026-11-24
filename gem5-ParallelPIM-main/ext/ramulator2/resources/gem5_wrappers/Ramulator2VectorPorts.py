from m5.SimObject import *
from m5.params import *
from m5.objects.AbstractMemory import *

class Ramulator2VectorPorts(AbstractMemory):
  type = "Ramulator2VectorPorts"
  cxx_class = "gem5::memory::Ramulator2VectorPorts"
  cxx_header = "mem/ramulator2/ramulator2_vector_ports.hh"

  ports = VectorResponsePort("Vector ports for receiving memory requests and sending responses")
  port_ranges = VectorParam.AddrRange([], "Address range served by each vector port")
  ramulator_config = Param.String("Ramulator 2 configuration (set automatically by ramulator.gem5.VectorPortMemory)")
