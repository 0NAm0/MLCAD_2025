import openroad as ord

def clone_buffer(design, inst_name):
    block = design.getBlock()
    orig  = block.findInst(inst_name)
    master = orig.getMaster()
    # create new instance next to original
    new_x = orig.getBBox().xMax() + 2
    new_y = orig.getBBox().yMin()
    db = design.getDb()
    cloned = ord.dbInst_create(block, master, f"{inst_name}_clone")
    cloned.setLocation(new_x, new_y)
    # wire: copy all nets
    for pin in orig.getITerms():
        cloned_pin = cloned.findITerm(pin.getMTerm().getConstName())
        cloned_pin.connect(pin.getNet())
    return cloned


def apply_gate_op(design, op_dict):
    if op_dict["op"] == "clone_buffer":
        clone_buffer(design, op_dict["inst"])
