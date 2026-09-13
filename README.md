PostgreSQL Database Management System
=====================================

This directory contains the source code distribution of the PostgreSQL
database management system.

PostgreSQL is an advanced object-relational database management system
that supports an extended subset of the SQL standard, including
transactions, foreign keys, subqueries, triggers, user-defined types
and functions.  This distribution also contains C language bindings.

Copyright and license information can be found in the file COPYRIGHT.

General documentation about this version of PostgreSQL can be found at
<https://www.postgresql.org/docs/18/>.  In particular, information
about building PostgreSQL from the source code can be found at
<https://www.postgresql.org/docs/18/installation.html>.

The latest version of this software, and related software, may be
obtained at <https://www.postgresql.org/download/>.  For more information
look at our web site located at <https://www.postgresql.org/>.

## Calico buffer manager (fork-specific)

This fork adds Calico as an alternative shared-buffer translation path. The
default `hashtable` implementation remains available. Calico maps each
relation fork to a compact identifier, then uses the block number to directly
index that relation's second-level translation array. The entries identify
buffer frames and track mapping state; PostgreSQL continues to manage the
shared buffer pool and its I/O and replacement policies. Each backend caches
its most recent relation-fork translation path in thread-local storage to
avoid repeating the shared relation lookup for nearby accesses.

The main settings are:

| Setting | Default | Effect |
| --- | --- | --- |
| `custom.buffer_manager_api` | `hashtable` | Selects `hashtable` or `calico`. Set this in `postgresql.conf`; restart PostgreSQL to apply the selection. |
| `calico.second_level_total_size_mb` | `1024` | Reserves shared memory for Calico's second-level translation arrays. Size it for the expected mapping footprint; changing the reservation requires a restart. |

`custom.enable_optimistic_buffer_reads` is also registered (default `off`),
but the current lookup paths do not read this setting, so changing it has no
runtime effect in this revision.
