import logging, time, sys, os


def setup_logging():
    handlers = [logging.StreamHandler(sys.stdout)]
    logpath = None
    if hasattr(sys.modules["__main__"], "__file__"):
        mainpath = os.path.abspath(sys.modules["__main__"].__file__)
        maindir = os.path.dirname(mainpath)
        main_body, _ = os.path.splitext(os.path.basename(mainpath))

        timestr = time.strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join(maindir, "logs")
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        logpath = os.path.join(logdir, "{}_{}.log".format(main_body, timestr))
        handlers.append(logging.FileHandler(logpath))

    # This is needed to get around fact that UDS creates
    # its own log file, which will preventa the one created
    # created above from actually being written to. There's
    # probably a smarter way to do this though...
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    log = logging.getLogger(__name__)
    if logpath:
        log.info("Setting up logger at {}".format(logpath))
    else:
        log.warn(
            "No main file to build a log relative to. Proceeding without stdout forwarding."
        )

    return log
