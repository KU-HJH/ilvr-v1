import re

from evaluation.schema import ContractError, ModelInput, ParsedAnswer, Simulation, VspTarget


WS = ' \t\r\n'
MARKER = re.compile(
    r'^[ \t]*(?:final[ \t]+answer(?:[ \t]+is)?|action[ \t]+plan)[ \t]*[:：]',
    re.I | re.M,
)


def extract_answer(raw_response: str, model_input: ModelInput) -> ParsedAnswer:
    if not isinstance(raw_response, str):
        raise ContractError('response must be a string')
    start = raw_response.rfind('\\boxed{')
    if start >= 0:
        begin, depth = start + len('\\boxed{'), 1
        for pos in range(begin, len(raw_response)):
            depth += (raw_response[pos] == '{') - (raw_response[pos] == '}')
            if depth == 0:
                value = raw_response[begin:pos].strip(WS)
                return ParsedAnswer(value, 'parsed' if value else 'empty', 'boxed')
        return ParsedAnswer('', 'invalid_format', 'boxed', 'unclosed_box')
    markers = list(MARKER.finditer(raw_response))
    value = raw_response[markers[-1].end():] if markers else raw_response
    value = value.strip(WS)
    return ParsedAnswer(value, 'parsed' if value else 'empty', 'marker' if markers else 'raw')


def validate_path(answer: ParsedAnswer) -> ParsedAnswer:
    if answer.status != 'parsed':
        return answer
    value = answer.value.upper()
    if not re.fullmatch(r'[UDLR ,\t\r\n]+', value) or any(
        not group.strip(WS) for group in value.split(',')
    ):
        return ParsedAnswer(answer.value, 'invalid_format', answer.method, 'invalid_actions')
    return ParsedAnswer(''.join(ch for ch in value if ch in 'UDLR'), 'parsed', answer.method)


def normalize_map(target: VspTarget) -> tuple[tuple[str, ...], ...]:
    if type(target.max_steps) is not int or target.max_steps <= 0:
        raise ContractError('max_steps must be a positive integer')
    grid = target.map_desc
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ContractError('map must be a nonempty rectangle')
    mapping = {0: 'F', 1: 'S', 2: 'G', -1: 'H'}
    result = []
    for row in grid:
        cells = []
        for cell in row:
            if type(cell) is int and cell in mapping:
                cells.append(mapping[cell])
            elif type(cell) is str and cell in ('F', 'S', 'G', 'H'):
                cells.append(cell)
            else:
                raise ContractError('unknown map cell')
        result.append(tuple(cells))
    flat = [cell for row in result for cell in row]
    if flat.count('S') != 1 or flat.count('G') != 1:
        raise ContractError('map requires exactly one start and goal')
    return tuple(result)


def compare_answer(answer: ParsedAnswer, target: VspTarget) -> Simulation:
    if answer.status != 'parsed' or not re.fullmatch('[UDLR]+', answer.value):
        raise ContractError('compare requires canonical parsed actions')
    grid = normalize_map(target)
    row, col = next(
        (r, c)
        for r, cells in enumerate(grid)
        for c, cell in enumerate(cells)
        if cell == 'S'
    )
    moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    for step, action in enumerate(answer.value, 1):
        dr, dc = moves[action]
        nr, nc = row + dr, col + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            row, col = nr, nc
        cell = grid[row][col]
        terminated = cell in ('G', 'H')
        truncated = step >= target.max_steps
        reward = 1.0 if cell == 'G' else 0.0
        if terminated or truncated:
            reason = 'goal' if cell == 'G' else 'hole' if cell == 'H' else 'step_limit'
            return Simulation(bool(reward), (row, col), step, reward, terminated, truncated, reason)
    return Simulation(False, (row, col), len(answer.value), 0.0, False, False, 'path_exhausted')
