import re
from .logger import get_logger

# ASTRA-Sim's per-iteration report, the one line of its stdout the frontend
# has to parse. Compiled once: parse_output runs on every handshake, and at
# 8 NPUs a 10-request run makes 337,786 of them.
_ITERATION_RE = re.compile(
    r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
    r"exposed communication (\d+) cycles."
)


class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        """Read ASTRA-Sim's stdout up to the "Waiting" prompt.

        Every line before the prompt is the iteration report; ASTRA-Sim used
        to interleave a per-tick "Checking ..." line per NPU, which made this
        loop 3.07M reads on a 10-request 8-NPU run against 675k now. See the
        ASTRA_SIM_TRACE_POLLING note in the analytical backend's main.cc.
        """
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            # For debugging
            # print(line, end='')
            if line == "":
                self._backend_gone(p, out)
            out.append(line)
        return out

    def _backend_gone(self, p, out):
        """The backend closed stdout mid-protocol: it crashed or exited.

        Without this the loop above reads "" forever at 100% CPU, which is
        indistinguishable from a long iteration from the outside.
        """
        rc = p.poll()
        if rc is None:
            try:
                rc = p.wait(timeout=5)
            except Exception:
                rc = None
        tail = "".join(l for l in out[-6:] if l).strip()
        raise RuntimeError(
            f"network backend exited (return code {rc}) before the next 'Waiting'; "
            f"last output: {tail[-400:]!r}")

    def check_end(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            line = p.stdout.readline()
            if line == "":
                self._backend_gone(p, out)
            out.append(line)
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_output(self, output):
        match = _ITERATION_RE.search(output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return