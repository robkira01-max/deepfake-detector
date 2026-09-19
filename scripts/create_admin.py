#!/usr/bin/env python3
"""Crée le premier utilisateur administrateur.

Usage :
    cd backend
    python ../scripts/create_admin.py
    python ../scripts/create_admin.py --username admin --email admin@example.com --password MonMotDePasse123!
    python ../scripts/create_admin.py --list
"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gestion des utilisateurs DeepfakeDetector")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--email", default="admin@deepfakedetector.ca")
    parser.add_argument("--password", default=None)
    parser.add_argument("--role", default="admin", choices=["admin", "analyst", "readonly"])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    from database import SessionLocal, init_db
    from models.user import User, UserRole
    from core.security import hash_password

    init_db()
    db = SessionLocal()

    if args.list:
        users = db.query(User).order_by(User.id).all()
        if not users:
            print("Aucun utilisateur.")
        else:
            print(f"{'ID':<4} {'Username':<20} {'Email':<30} {'Rôle':<10} {'Actif':<6} {'Dernière connexion'}")
            print("-" * 90)
            for u in users:
                last = u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "—"
                print(f"{u.id:<4} {u.username:<20} {u.email:<30} {u.role.value:<10} {str(u.is_active):<6} {last}")
        db.close()
        return

    password = args.password
    if not password:
        print(f"Création de l'utilisateur '{args.username}' (rôle: {args.role})")
        password = getpass.getpass("Mot de passe (min 12 caractères) : ")
        confirm = getpass.getpass("Confirmer : ")
        if password != confirm:
            print("ERREUR : Les mots de passe ne correspondent pas.", file=sys.stderr)
            sys.exit(1)

    if len(password) < 12:
        print("ERREUR : Minimum 12 caractères.", file=sys.stderr)
        sys.exit(1)

    existing = db.query(User).filter(User.username == args.username).first()
    if existing:
        overwrite = input(f"L'utilisateur '{args.username}' existe. Réinitialiser le mot de passe ? [o/N] ").strip().lower()
        if overwrite == "o":
            existing.hashed_password = hash_password(password)
            db.commit()
            print(f"Mot de passe de '{args.username}' réinitialisé.")
        db.close()
        return

    user = User(
        username=args.username,
        email=args.email,
        hashed_password=hash_password(password),
        role=UserRole[args.role],
    )
    db.add(user)
    db.commit()
    print(f"✅ Utilisateur '{args.username}' ({args.role}) créé avec succès.")
    print(f"   API : http://localhost:8000/docs")
    db.close()


if __name__ == "__main__":
    main()
