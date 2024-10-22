from sofos._2048.game.classes import Board

try:
    # for Windows-based systems
    import msvcrt

    def getchar():
        return msvcrt.getch()

except ImportError:
    # for POSIX-based systems (with termios & tty support)
    import sys
    import termios
    import tty  # raises ImportError if unsupported

    def getchar():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            answer = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return answer


def main():
    board = Board()
    board.add_random_tiles(2)

    move_counter = 0
    move = None
    move_result = False

    while True:
        print(
            f"Number of successful moves:{move_counter}, "
            f"Last move attempted:{move}:, "
            f"Move status:{move_result}"
        )
        print(board)
        key = getchar()

        if key == b"q" or key == "q":
            quit()

        if key == b"w" or key == "w":
            move = "UP"
        elif key == b"a" or key == "a":
            move = "LEFT"
        elif key == b"s" or key == "s":
            move = "DOWN"
        elif key == b"d" or key == "d":
            move = "RIGHT"
        else:
            move = None

        if move is not None:
            move_result = board.make_move(move)
            if move_result:
                board.add_random_tiles(1)
                move_counter = move_counter + 1


if __name__ == "__main__":
    main()
