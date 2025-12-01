#!/usr/bin/env python3
"""
Convert CHARMM CRD files to PDB format.

CHARMM CRD format:
- Title lines (starting with *)
- Number of atoms line
- Atom lines with format: ATOMNO RESNO RES RESID ATOMNAME X Y Z SEGID RESID Weight

PDB ATOM format:
ATOM  serial  name resName chainID resSeq    x       y       z     occupancy tempFactor
"""

import sys
import argparse
import glob
from pathlib import Path

def parse_crd_file(crd_file):
    """Parse a CHARMM CRD file and extract atom information."""
    atoms = []
    
    with open(crd_file, 'r') as f:
        lines = f.readlines()
    
    # Skip title lines (lines starting with *)
    i = 0
    while i < len(lines) and lines[i].strip().startswith('*'):
        i += 1
    
    # Read number of atoms
    if i < len(lines):
        natoms = int(lines[i].split()[0])
        i += 1
    else:
        raise ValueError("Could not find number of atoms in CRD file")
    
    # Parse atom lines
    for line in lines[i:i+natoms]:
        parts = line.split()
        if len(parts) >= 9:
            atom_data = {
                'serial': int(parts[0]),
                'resSeq': int(parts[1]),
                'resName': parts[2],
                'atomName': parts[3],
                'x': float(parts[4]),
                'y': float(parts[5]),
                'z': float(parts[6]),
                'segID': parts[7] if len(parts) > 7 else '',
                'chainID': 'A'#parts[8][0] if len(parts) > 8 and parts[8] else 'A'
            }
            atoms.append(atom_data)
    
    return atoms


def write_pdb_file(atoms, pdb_file):
    """Write atoms to PDB format file."""
    with open(pdb_file, 'w') as f:
        # Write header
        f.write("REMARK   Generated from CHARMM CRD file\n")
        
        # Write ATOM records
        for atom in atoms:
            # PDB format specification:
            # ATOM  serial name resName chainID resSeq    x       y       z     occupancy tempFactor
            pdb_line = (
                f"ATOM  {atom['serial']:>5d} {atom['atomName']:<4s} "
                f"{atom['resName']:<3s} {atom['chainID']:>1s}"
                f"{atom['resSeq']:>4d}    "
                f"{atom['x']:>8.3f}{atom['y']:>8.3f}{atom['z']:>8.3f}"
                f"  1.00  0.00           \n"
            )
            f.write(pdb_line)
        
        # Write END record
        f.write("END\n")


def convert_crd_to_pdb(crd_file, pdb_file):
    """Convert CHARMM CRD file to PDB format."""
    try:
        atoms = parse_crd_file(crd_file)
        write_pdb_file(atoms, pdb_file)
        print(f"Successfully converted {crd_file} to {pdb_file}")
        print(f"Total atoms written: {len(atoms)}")
    except Exception as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        raise e

def main():
    crd_files = sorted(glob.glob('popc/*/*.crd'))
    for crd_file in crd_files:
        outname = str(Path(crd_file).with_suffix('.pdb'))
        convert_crd_to_pdb(crd_file, outname)

if __name__ == '__main__':
    main()
