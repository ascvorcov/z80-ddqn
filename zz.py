import zipfile

mem = [11,22,33]*1000000
print('saving memory snapshot')
with zipfile.ZipFile('./output/logs/memory.zip', 'w', compression=zipfile.ZIP_DEFLATED) as memzip:
    for i,record in enumerate(mem):
        cs = bytearray(10)
        ns = bytearray(10)
        ac = 1
        rw = record
        t = 1 if record % 2 == 0 else 0
        memzip.writestr(f'cs{i}-{ac}-{rw}-{t}.bin', cs)
        memzip.writestr(f'ns{i}-{ac}-{rw}-{t}.bin', ns)
