#!/usr/bin/python

import model
import export_to_gcp


def main():
    model.main()
    export_to_gcp.main()


if __name__ == '__main__':
    main()
