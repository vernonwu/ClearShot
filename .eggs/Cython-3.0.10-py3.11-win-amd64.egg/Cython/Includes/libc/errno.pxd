# 7.5 Errors <errno.h>

cdef extern from "<errno.h>" nogil:
    enum:
        EPERM
        ENOENT
        ESRCH
        EINTR
        EIO
        ENXIO
        E2BIG
        ENOEXEC
        EBADF
        ECHILD
        EAGAIN
        ENOMEM
        EACCES
        EFAULT
        ENOTBLK
        EBUSY
        EEXIST
        EXDEV
        ENODEV
        ENOTDIR
        EISDIR
        EINVAL
        ENFILE
        EMFILE
        ENOTTY
        ETXTBSY
        EFBIG
        ENOSPC
        ESPIPE
        EROFS
        EMLINK
        EPIPE
        EDOM
        ERANGE
        EDEADLOCK
        ENAMETOOLONG
        ENOLCK
        ENOSYS
        ENOTEMPTY
        ELOOP
        ENOMSG
        EIDRM
        ECHRNG
        EL2NSYNC
        EL3HLT
        EL3RST
        ELNRNG
        EUNATCH
        ENOCSI
        EL2HLT
        EBADE
        EBADR
        EXFULL
        ENOANO
        EBADRQC
        EBADSLT
        EBFONT
        ENOSTR
        ENODATA
        ENOATTR
        ETIME
        ENOSR
        ENONET
        ENOPKG
        EREMOTE
        ENOLINK
        EADV
        ESRMNT
        ECOMM
        EPROTO
        EMULTIHOP
        EDOTDOT
        EBADMSG
        EOVERFLOW
        ENOTUNIQ
        EBADFD
        EREMCHG
        ELIBACC
        ELIBBAD
        ELIBSCN
        ELIBMAX
        ELIBEXEC
        EILSEQ
        ERESTART
        ESTRPIPE
        EUSERS
        ENOTSOCK
        EDESTADDRREQ
        EMSGSIZE
        EPROTOTYPE
        ENOPROTOOPT
        EPROTONOSUPPORT
        ESOCKTNOSUPPORT
        EOPNOTSUPP
        EPFNOSUPPORT
        EAFNOSUPPORT
        EADDRINUSE
        EADDRNOTAVAIL
        ENETDOWN
        ENETUNREACH
        ENETRESET
        ECONNABORTED
        ECONNRESET
        ENOBUFS
        EISCONN
        ENOTCONN
        ESHUTDOWN
        ETOOMANYREFS
        ETIMEDOUT
        ECONNREFUSED
        EHOSTDOWN
        EHOSTUNREACH
        EALREADY
        EINPROGRESS
        ESTALE
        EUCLEAN
        ENOTNAM
        ENAVAIL
        EISNAM
        EREMOTEIO
        EDQUOT

    int errno
